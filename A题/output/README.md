# output 目录说明

## figures（论文图，保留）

| 文件 | 对应章节 |
|------|------|
| problem1_comparison.png | 问题1 四方法对比 |
| problem2_comparison.png | 问题2 预测vs真值 |
| problem2_weights.png | 问题2 权重对比 |
| problem3_feature_discrimination.png | 问题3 特征判别力 |

## figures（备选，待定）

| 文件 | 内容 |
|------|------|
| problem3_dendrogram.png | 聚类谱系图 |
| problem3_pca_scatter.png | PCA散点图 |
| problem3_feature_dist.png | 双阈值散点 |
| problem3_compensation_heatmap.png | 补偿热力图 |

## results（论文用，保留）

| 文件 | 内容 |
|------|------|
| problem3_params.json | 问题3在线规则固定参数 |
| problem3_evaluation_summary.csv | 三种口径评价汇总 |
| problem3_compensation_params.csv | A/B类×流量点补偿系数 |
| problem3_silhouette.csv | K=2-5轮廓系数 |
| problem3_zero_bias_submission.csv | 最终提交文件 |
| problem3_d0_aligned_submission.csv | 备选提交文件 |

## results（诊断表，可删除）

| 文件 | 内容 |
|------|------|
| problem1_error_summary/by_flow.csv | 问题1逐窗口误差 |
| problem2_cv_by_date/groups/results.csv | 问题2诊断 |
| problem3_classification.csv | 每窗口在线分类标签 |
| problem3_cluster_assignment.csv | D1-D8聚类标签 |
| problem3_compensated_results.csv | 每窗口补偿前后误差 |
| problem3_detection_flags.csv | 每窗口检测标记 |
| problem3_feature_discrimination.csv | 各特征Cohen's d值 |
