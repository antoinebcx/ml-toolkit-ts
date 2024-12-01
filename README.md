# ml-toolkit-ts

A TypeScript toolkit for machine learning inference and operations.

Run your XGBoost models trained with [ElectronML](https://github.com/antoinebcx/ElectronML) directly in TypeScript/JavaScript!

## Installation

Install everything:
```bash
npm install ml-toolkit-ts
```

Or install specific packages:
```bash
npm install @ml-toolkit-ts/xgboost
npm install @ml-toolkit-ts/preprocessing
```

## Usage

### 1. Load your inference package

Use the JSON file you created with [ElectronML](https://github.com/antoinebcx/ElectronML).

```typescript
// Type definition for inference package
interface InferencePackage {
  model: any;                                    
  preprocessing_metadata: any;                   
  feature_names: string[];                       
  class_mapping: Record<number, string>;         
  isRegression: boolean;                         
}

// Browser: using async/await
async function loadModel() {
  const inferencePackage = await fetch('model/inference_package.json')
    .then(response => response.json());
    // Use inferencePackage...
}

// Node.js
import * as fs from 'fs';
const inferencePackage = JSON.parse(
  fs.readFileSync('model/inference_package.json', 'utf-8')
);
```

### 2. Initialize predictor and preprocessor

```typescript
// Using the complete package
import { XGBoostPredictor, DataPreprocessor } from 'ml-toolkit-ts';

// Or import specific packages
import { XGBoostPredictor } from '@ml-toolkit-ts/xgboost';
import { DataPreprocessor } from '@ml-toolkit-ts/preprocessing';


const predictor = new XGBoostPredictor(JSON.stringify(inferencePackage.model));
const preprocessor = new DataPreprocessor(JSON.stringify(inferencePackage.preprocessing_metadata));
```

### 3. Make predictions

**For classification:**
```typescript
const inputValues = {
  age: "25",
  income: "50000",
  category: "A"
};

const transformedFeatures = preprocessor.transform(inputValues);
const predictedClass = predictor.predict(transformedFeatures);
const probabilities = predictor.predict_proba(transformedFeatures);

console.log('Predicted class:', inferencePackage.class_mapping[predictedClass]);
console.log('Probabilities:', probabilities);
// Output:
// Predicted class: high_risk
// Probabilities: [0.15, 0.85]  // 15% low_risk, 85% high_risk
```

**For regression:**
```typescript
const inputValues = {
  sqft: "1500",
  bedrooms: "3",
  location: "urban"
};

const transformedFeatures = preprocessor.transform(inputValues);
const prediction = predictor.predict(transformedFeatures);

console.log('Predicted price:', prediction);
// Output:
// Predicted price: 450000
```