# RHALE: Robust and Heterogeneity-aware Accumulated Local Effects

Paper accepted at [ECAI 2023](https://ecai2023.eu/)


Authors:

* Vasilis Gkolemis ([vgkolemis@athenarc.gr](), [gkolemis@hua.gr]())
* Theodore Dalamagas ([dalamag@athenarc.gr]())
* Eirini Ntoutsi ([eirini.ntoutsi@unibw.de]())
* Christos Diou ([cdiou@hua.gr]())

Abstract:

Accumulated Local Effects (ALE) is a widely-used explainability method for isolating the average effect of a feature on the
output, because it handles cases with correlated features well. However, it has two limitations. First, it does not quantify the deviation of
instance-level (local) effects from the average (global) effect, known
as heterogeneity. Second, for estimating the average effect, it partitions the feature domain into user-defined, fixed-sized bins, where
different bin sizes may lead to inconsistent ALE estimations. To address these limitations, we propose Robust and Heterogeneity-aware
ALE (RHALE). RHALE quantifies the heterogeneity by considering
the standard deviation of the local effects and automatically determines an optimal variable-size bin-splitting. In this paper, we prove
that to achieve an unbiased approximation of the standard deviation
of local effects within each bin, bin splitting must follow a set of
sufficient conditions. Based on these conditions, we propose an algorithm that automatically determines the optimal partitioning, bal-
ancing the estimation bias and variance. Through evaluations on synthetic and real datasets, we demonstrate the superiority of RHALE
compared to other methods, including the advantages of automatic
bin splitting, especially in cases with correlated features.