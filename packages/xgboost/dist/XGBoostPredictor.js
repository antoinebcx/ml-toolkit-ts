"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.XGBoostPredictor = void 0;
class XGBoostPredictor {
    constructor(modelJson) {
        var _a, _b, _c, _d, _e;
        try {
            this.model = JSON.parse(modelJson);
            if (!((_c = (_b = (_a = this.model) === null || _a === void 0 ? void 0 : _a.learner) === null || _b === void 0 ? void 0 : _b.objective) === null || _c === void 0 ? void 0 : _c.name)) {
                throw new Error("Invalid model format: missing objective");
            }
            this.objective = this.model.learner.objective.name;
            if (this.objective.startsWith('multi:')) {
                this.numClasses = parseInt(((_d = this.model.learner.objective.softmax_multiclass_param) === null || _d === void 0 ? void 0 : _d.num_class) || '2');
                this.isClassification = true;
            }
            else if (this.objective.startsWith('binary:')) {
                this.numClasses = 2;
                this.isClassification = true;
            }
            else {
                this.numClasses = 1;
                this.isClassification = false;
            }
            const firstTree = this.model.learner.gradient_booster.model.trees[0];
            if (!firstTree) {
                throw new Error("Model contains no trees");
            }
            this.numFeatures = parseInt(firstTree.tree_param.num_feature);
            this.treeCache = new Map();
            this.baseScore = parseFloat(((_e = this.model.learner.attributes) === null || _e === void 0 ? void 0 : _e.base_score) || '0.5');
        }
        catch (e) {
            throw new Error(`Failed to initialize XGBoost model: ${e instanceof Error ? e.message : 'Unknown error'}`);
        }
    }
    getCacheKey(treeIndex, features) {
        return `${treeIndex}_${features.join('_')}`;
    }
    traverseTreeWithCache(treeIndex, tree, features) {
        const cacheKey = this.getCacheKey(treeIndex, features);
        if (this.treeCache.has(cacheKey)) {
            return this.treeCache.get(cacheKey);
        }
        const result = this.traverseTree(tree, features);
        this.treeCache.set(cacheKey, result);
        return result;
    }
    traverseTree(tree, features) {
        let nodeIndex = 0;
        const { left_children, right_children, split_indices, split_conditions } = tree;
        while (true) {
            if (left_children[nodeIndex] === -1) {
                return tree.base_weights[nodeIndex];
            }
            const featureIndex = split_indices[nodeIndex];
            if (featureIndex >= features.length) {
                throw new Error(`Invalid feature index: ${featureIndex}`);
            }
            nodeIndex = features[featureIndex] <= split_conditions[nodeIndex]
                ? left_children[nodeIndex]
                : right_children[nodeIndex];
        }
    }
    sigmoid(x) {
        if (x < -40)
            return 0;
        if (x > 40)
            return 1;
        return 1 / (1 + Math.exp(-x));
    }
    softmax(scores) {
        const maxScore = Math.max(...scores);
        const expScores = new Float64Array(scores.length);
        let sumExp = 0;
        for (let i = 0; i < scores.length; i++) {
            const exp = scores[i] - maxScore < -40 ? 0 : Math.exp(scores[i] - maxScore);
            expScores[i] = exp;
            sumExp += exp;
        }
        if (sumExp === 0)
            sumExp = 1;
        for (let i = 0; i < expScores.length; i++) {
            expScores[i] /= sumExp;
        }
        return Array.from(expScores);
    }
    predict(features) {
        this.validateFeatures(features);
        if (!this.isClassification) {
            return this.predictRaw(features)[0];
        }
        if (this.numClasses === 2) {
            const probs = this.predict_proba(features);
            return probs[1] >= 0.5 ? 1 : 0;
        }
        else {
            const margins = this.predictRaw(features);
            return margins.indexOf(Math.max(...margins));
        }
    }
    predictRaw(features) {
        const trees = this.model.learner.gradient_booster.model.trees;
        const treeInfo = this.model.learner.gradient_booster.model.tree_info;
        if (this.numClasses === 1 || this.numClasses === 2) {
            let sum = this.baseScore;
            for (let i = 0; i < trees.length; i++) {
                sum += this.traverseTreeWithCache(i, trees[i], features);
            }
            return [sum];
        }
        else {
            const margins = new Array(this.numClasses).fill(0);
            for (let i = 0; i < trees.length; i++) {
                const classIndex = treeInfo[i];
                margins[classIndex] += this.traverseTreeWithCache(i, trees[i], features);
            }
            return margins;
        }
    }
    predict_proba(features) {
        if (!this.isClassification) {
            throw new Error("predict_proba is only available for classification tasks");
        }
        this.validateFeatures(features);
        const margins = this.predictRaw(features);
        if (this.numClasses === 2) {
            const probability = this.objective === 'binary:hinge'
                ? (margins[0] > 0 ? 1 : 0)
                : this.sigmoid(margins[0]);
            return [1 - probability, probability];
        }
        return this.softmax(margins);
    }
    validateFeatures(features) {
        if (!Array.isArray(features)) {
            throw new Error("Features must be an array");
        }
        if (features.length !== this.numFeatures) {
            throw new Error(`Expected ${this.numFeatures} features, got ${features.length}`);
        }
        if (!features.every(f => typeof f === 'number' && !isNaN(f))) {
            throw new Error("All features must be valid numbers");
        }
    }
    getFeatureImportances() {
        const importances = new Array(this.numFeatures).fill(0);
        const trees = this.model.learner.gradient_booster.model.trees;
        for (const tree of trees) {
            for (let i = 0; i < tree.split_indices.length; i++) {
                const featureIndex = tree.split_indices[i];
                if (featureIndex >= 0 && featureIndex < this.numFeatures) {
                    importances[featureIndex]++;
                }
            }
        }
        const sum = importances.reduce((a, b) => a + b, 0);
        return importances.map(v => sum > 0 ? v / sum : 0);
    }
    getModelInfo() {
        return {
            numClasses: this.numClasses,
            numFeatures: this.numFeatures,
            numTrees: this.model.learner.gradient_booster.model.trees.length,
            objective: this.objective,
            featureNames: this.model.learner.feature_names || [],
            baseScore: this.baseScore,
            isClassification: this.isClassification
        };
    }
    clearCache() {
        this.treeCache.clear();
    }
}
exports.XGBoostPredictor = XGBoostPredictor;
