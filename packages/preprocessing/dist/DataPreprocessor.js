"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DataPreprocessor = void 0;
class DataPreprocessor {
    constructor(metadataJson) {
        this.metadata = JSON.parse(metadataJson);
    }
    standardScaleValue(value, mean, scale) {
        return (value - mean) / scale;
    }
    minMaxScaleValue(value, min, scale) {
        return (value - min) / scale;
    }
    transform(input) {
        const result = [];
        const missingFeatures = this.metadata.features.filter(f => !(f in input));
        if (missingFeatures.length > 0) {
            throw new Error(`Missing required features: ${missingFeatures.join(', ')}`);
        }
        for (const feature of this.metadata.features) {
            const value = input[feature];
            if (feature in this.metadata.categorical_features) {
                const mapping = this.metadata.categorical_features[feature];
                const strValue = String(value);
                let encodedValue;
                if (strValue in mapping) {
                    encodedValue = mapping[strValue];
                }
                else {
                    encodedValue = mapping[Object.keys(mapping)[0]];
                }
                result.push(encodedValue);
            }
            else if (feature in this.metadata.numeric_features) {
                const numValue = Number(value);
                if (isNaN(numValue)) {
                    throw new Error(`Invalid numeric value for feature ${feature}: ${value}`);
                }
                const params = this.metadata.numeric_features[feature];
                if (this.metadata.scaling_method === 'standard') {
                    const scaled = this.standardScaleValue(numValue, params.mean, params.scale);
                    result.push(scaled);
                }
                else {
                    const scaled = this.minMaxScaleValue(numValue, params.min, params.scale);
                    result.push(scaled);
                }
            }
        }
        return result;
    }
    getFeatureInfo() {
        return {
            featureNames: this.metadata.features,
            categoricalFeatures: Object.fromEntries(Object.entries(this.metadata.categorical_features).map(([feature, mapping]) => [feature, Object.keys(mapping)])),
            numericFeatures: Object.keys(this.metadata.numeric_features)
        };
    }
}
exports.DataPreprocessor = DataPreprocessor;
