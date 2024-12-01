import { PipelineMetadata } from './types';

export class DataPreprocessor {
  private metadata: PipelineMetadata;

  constructor(metadataJson: string) {
    this.metadata = JSON.parse(metadataJson);
  }

  private standardScaleValue(value: number, mean: number, scale: number): number {
    return (value - mean) / scale;
  }

  private minMaxScaleValue(value: number, min: number, scale: number): number {
    return (value - min) / scale;
  }

  transform(input: Record<string, string | number>): number[] {
    const result: number[] = [];

    const missingFeatures = this.metadata.features.filter(f => !(f in input));
    if (missingFeatures.length > 0) {
      throw new Error(`Missing required features: ${missingFeatures.join(', ')}`);
    }

    for (const feature of this.metadata.features) {
      const value = input[feature];

      if (feature in this.metadata.categorical_features) {
        const mapping = this.metadata.categorical_features[feature];
        const strValue = String(value);
        
        let encodedValue: number;
        if (strValue in mapping) {
          encodedValue = mapping[strValue];
        } else {
          encodedValue = mapping[Object.keys(mapping)[0]];
        }
        result.push(encodedValue);

      } else if (feature in this.metadata.numeric_features) {
        const numValue = Number(value);
        if (isNaN(numValue)) {
          throw new Error(`Invalid numeric value for feature ${feature}: ${value}`);
        }

        const params = this.metadata.numeric_features[feature];
        if (this.metadata.scaling_method === 'standard') {
          const scaled = this.standardScaleValue(
            numValue,
            params.mean!,
            params.scale!
          );
          result.push(scaled);
        } else {
          const scaled = this.minMaxScaleValue(
            numValue,
            params.min!,
            params.scale!
          );
          result.push(scaled);
        }
      }
    }

    return result;
  }

  getFeatureInfo(): {
    featureNames: string[];
    categoricalFeatures: Record<string, string[]>;
    numericFeatures: string[];
  } {
    return {
      featureNames: this.metadata.features,
      categoricalFeatures: Object.fromEntries(
        Object.entries(this.metadata.categorical_features).map(
          ([feature, mapping]) => [feature, Object.keys(mapping)]
        )
      ),
      numericFeatures: Object.keys(this.metadata.numeric_features)
    };
  }
}