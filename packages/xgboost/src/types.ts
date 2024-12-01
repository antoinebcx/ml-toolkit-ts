export interface TreeParam {
    num_deleted: string;
    num_feature: string;
    num_nodes: string;
    size_leaf_vector: string;
}
  
export interface TreeNode {
    id?: number;
    split_indices: number[];
    split_conditions: number[];
    split_type: number[];
    left_children: number[];
    right_children: number[];
    parents: number[];
    base_weights: number[];
    tree_param: TreeParam;
}
  
export interface GradientBooster {
    model: {
      trees: TreeNode[];
      tree_info: number[];
    };
}
  
export interface ObjectiveParam {
    name: string;
    softmax_multiclass_param?: {
      num_class: string;
    };
}
  
export interface XGBoostModel {
    learner: {
      objective: ObjectiveParam;
      gradient_booster: GradientBooster;
      attributes?: Record<string, string>;
      feature_names?: string[];
    };
}