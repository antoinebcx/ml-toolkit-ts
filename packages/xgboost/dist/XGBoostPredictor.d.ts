export declare class XGBoostPredictor {
    private model;
    private numClasses;
    private numFeatures;
    private treeCache;
    private baseScore;
    private objective;
    private isClassification;
    constructor(modelJson: string);
    private getCacheKey;
    private traverseTreeWithCache;
    private traverseTree;
    private sigmoid;
    private softmax;
    predict(features: number[]): number;
    private predictRaw;
    predict_proba(features: number[]): number[];
    private validateFeatures;
    getFeatureImportances(): number[];
    getModelInfo(): Record<string, any>;
    clearCache(): void;
}
