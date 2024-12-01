export interface PipelineMetadata {
    features: string[];
    categorical_features: {
        [feature: string]: { [category: string]: number };
    };
    numeric_features: {
        [feature: string]: {
            mean?: number;
            scale?: number;
            min?: number;
        };
    };
    scaling_method: 'standard' | 'minmax';
}