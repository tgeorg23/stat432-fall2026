
---
name: svm-fitting
description: Fit and compare SVM classification models using cross-validation. Only use when explicitly requested.
---

# SVM Model Fitting

Only use this skill when `svm-fitting` is specifically requested.

1. Identify the response variable and predictors. If the target is unclear, ask the user before continuing.
2. This skill is for classification problems, not regression.
3. Use a random seed of 2026 and set aside 25% of the data for testing. Stratify the split when possible and don't use the test data until the end.
4. Use 5-fold stratified cross-validation on the training data. Keep the folds the same for every model. If a class has too few observations, reduce the number of folds.
5. Handle missing values, categorical variables, and scaling within a pipeline so that preprocessing is fitted separately in each training fold.
6. Compare linear and RBF SVMs. Also include a majority-class classifier as a baseline.
7. For linear SVM, try C values of 0.1, 1, and 10. For RBF SVM, use the same C values and gamma values of "scale", 0.01, 0.1, and 1.
8. Use average CV classification error to compare models. If the classes are highly imbalanced, choose a more suitable metric and explain why.
9. Calculate the standard error of each model's CV score. Find the model with the lowest average error, then use the one-standard-error rule to choose a simpler model if its error is close enough.
10. Prefer linear SVMs over RBF SVMs when their results are similar. For models with the same kernel, prefer smaller C values. Use the majority-class model for comparison, not as the final SVM.
11. Refit the selected model using all the training data. Evaluate it on the test set along with the best linear SVM, best RBF SVM, and majority-class baseline. Don't use these results to change the model.
12. Report the CV errors, standard errors, chosen model, and final test results. Include the test sample size and an approximate standard error for the test error rate.
13. Explain whether the selected SVM actually performed noticeably better than simpler models. Don't claim that small differences prove one model is better.
14. Include the preprocessing steps, train/test split, CV settings, selected parameters, and any limitations. Provide code so the analysis can be reproduced.
15. Keep the report straightforward. Don't make up results, include unnecessary table index columns, or add dataset-specific information to this skill.