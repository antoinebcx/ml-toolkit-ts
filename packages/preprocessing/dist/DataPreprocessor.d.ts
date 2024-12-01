export declare class DataPreprocessor {
    private metadata;
    constructor(metadataJson: string);
    private standardScaleValue;
    private minMaxScaleValue;
    transform(input: Record<string, string | number>): number[];
    getFeatureInfo(): {
        featureNames: string[];
        categoricalFeatures: Record<string, string[]>;
        numericFeatures: string[];
    };
}
