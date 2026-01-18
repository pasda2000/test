# 1. Introduction to Data Mining, Task Types, Input

## 1.1 Motivation and Application Areas

In modern society, an enormous amount of data is generated automatically through digital systems. Examples include shopping transactions, banking operations, mobile phone usage, sensor measurements in vehicles, medical examinations, and online activity. Storing this digital data is no longer a major technical problem, as storage has become cheap and widely available.

However, **understanding the data** is much more difficult than storing it. Raw data by itself does not provide insight. The real value lies in discovering *patterns*, *regularities*, and *relationships* hidden within the data. These patterns can support better decisions, reveal previously unknown connections, or improve existing processes.

The aim of **Data Mining** is to identify such patterns automatically or semi-automatically from large datasets.

Typical application areas include:
- **Banking**: fraud detection, credit scoring
- **Marketing**: customer segmentation, targeted advertising
- **Retail**: market basket analysis, recommendation systems
- **Telecommunications**: churn prediction
- **Insurance**: risk assessment
- **Healthcare**: discovering relationships between diseases, treatments, and lifestyle factors

---

## 1.2 Definition of Knowledge Discovery (Fayyad)

According to Fayyad et al. (1996), Knowledge Discovery in Databases (KDD) is defined as:

> *The non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data.*

Each component of the definition is important:

- **Process**: KDD is not a single step, but a multi-step, iterative procedure involving data preparation, model building, and evaluation. The results of later steps may require revisiting earlier ones.

- **Non-trivial**: The discovered patterns are not obvious and cannot be obtained by simple queries or basic statistics alone.

- **Valid**: Patterns must hold on new, unseen data with some degree of certainty. Overfitted or accidental patterns are not acceptable.

- **Novel**: The patterns should represent new knowledge, not something already known beforehand.

- **Potentially useful**: The patterns should be actionable or at least support decision-making.

- **Understandable**: The results should provide human insight. Even accurate models are of limited value if they cannot be interpreted or explained.

In this context:
- **Pattern** means a description or model that captures regularities in a subset of the data.
- **Data** refers to a collection of facts, typically represented as instances stored in a database or file.

---

## 1.3 The KDD Process

KDD stands for **Knowledge Discovery in Databases**. It is a structured, goal-driven, and iterative process consisting of the following main steps:

1. **Acquiring background domain knowledge**  
   Understanding the application domain, the meaning of attributes, and the goals of the analysis.

2. **Setting goals**  
   Defining what kind of patterns or predictions are sought and what success means.

3. **Data selection, reduction, and cleansing**  
   Choosing relevant data, handling missing values, correcting errors, removing duplicates, and possibly reducing dimensionality.

4. **Model selection and data mining**  
   Choosing appropriate data mining methods (e.g. classification, clustering) and learning models from the data.

5. **Interpretation and evaluation**  
   Interpreting the discovered patterns, evaluating their usefulness and validity, and deciding whether further iterations are necessary.

The process is **iterative**: insights gained at later stages often require going back and modifying earlier steps.

---

## 1.4 Goals of Data Mining

Data mining has two fundamental and complementary goals:

### Description
The goal of description is to find **human-interpretable patterns** that help understand the structure of the data. Examples include rules, decision trees, or clusters. Descriptive models are often referred to as *white-box models* because their internal logic can be inspected and explained.

### Prediction
The goal of prediction is to **predict unknown or future values**. Predictive models are used to support, automate, or improve decision-making. In many cases, prediction accuracy is more important than interpretability.

---

## 1.5 Major Classes of Data Mining Tasks

The main task types in data mining are the following:

### Classification
Classification is a **supervised learning** task. The data contains a special attribute called the *class*, which is nominal (categorical). The goal is to learn a model that assigns class labels to previously unseen instances.

### Numeric Prediction (Regression)
Numeric prediction, often called **regression**, is also supervised learning. The difference from classification is that the target attribute is numeric rather than nominal. The goal is to predict a numeric value for new instances.

### Clustering
Clustering is an **unsupervised learning** task. There is no predefined class attribute. The goal is to divide instances into groups based on similarity. The number of clusters and the similarity measure are not inherently defined and must be chosen.

### Association Rule Mining
The goal of association rule mining is to discover relationships of the form:

*If a set of attributes occurs in a record, then another attribute is likely to occur as well.*

There is no distinguished class attribute, and attributes are typically Boolean (present or not present). This is an unsupervised task.

---

## 1.6 Related Fields

Data mining is an interdisciplinary field that integrates ideas from several areas:

- **Statistics**: provides mathematical foundations and hypothesis testing, but is traditionally more theory-driven.
- **Machine Learning**: focuses on learning from experience and improving performance automatically, often emphasizing scalability and real-world data.
- **Databases**: provide efficient data storage, retrieval, and preprocessing capabilities.
- **Artificial Intelligence and Data Visualization**: contribute reasoning methods and tools for human understanding.

---

## 1.7 Input for Data Mining

### Instances
The input to a data mining algorithm is typically a **dataset**, represented as a single flat table. Each row corresponds to an **instance**, which is an independent example to be classified, predicted, associated, or clustered.

### Attributes
Each instance is described by a fixed, predefined set of **attributes**. Attributes represent measurable or observable properties of the instances.

---

## 1.8 Types of Attributes (Levels of Measurement)

### Nominal Attributes
Nominal attributes consist of distinct symbols or labels. There is no ordering or distance between values. Only equality or inequality can be tested.

Examples: gender, marital status.

### Ordinal Attributes
Ordinal attributes impose an order on values, but distances between values are not defined. Arithmetic operations are not meaningful.

Examples: small < medium < large.

### Interval Attributes
Interval attributes are ordered and measured in equal units. Differences between values are meaningful, but there is no true zero point.

Example: calendar year.

### Ratio Attributes
Ratio attributes have a true zero point and allow all mathematical operations.

Examples: age, height, weight.

Most data mining algorithms internally handle only **nominal** and **numeric** attributes, so transformations between data types are often necessary.

---

## 1.9 Transformation Between Data Types

Ordinal attributes can be transformed into multiple Boolean attributes. For example, an attribute with values {cold, medium, hot} can be encoded as:
- at_least_medium
- at_least_hot

This allows algorithms that only support nominal or numeric attributes to process ordinal data.

---

## 1.10 Missing and Inaccurate Data

Data may contain **missing values** or inaccuracies due to:
- measurement errors
- typographical errors
- malfunctioning equipment
- changes in experimental design
- data integration from multiple sources

A missing value may be unknown, unrecorded, or irrelevant. In some domains, missingness itself can be informative (e.g. a missing medical test), but most algorithms assume that missing values are simply unknown and require special handling or encoding.

---

### End of Topic 1

# 2. 1R, ID3, C4.5 Tree Learning Algorithms

## 2.1 The 1R (One Rule) Algorithm

### Form of Knowledge Representation
The 1R algorithm produces a **one-level decision tree**, also known as a *decision stump*. The learned model is equivalent to a set of classification rules that all test **the same single attribute**. Although all attributes are examined during learning, only one attribute is used during prediction.

### Basic Assumptions
The basic version of the 1R algorithm assumes:
- attributes are **nominal**,
- there are **no missing values** in the data.

Extensions exist to handle numeric attributes and missing values, but the basic idea remains unchanged.

### Learning Algorithm
The learning procedure evaluates each attribute independently and selects the one that yields the lowest classification error.

For each attribute in the dataset:
1. Create one branch (rule) for each possible value of the attribute.
2. For each branch, count how many training instances belong to each class.
3. Assign the **most frequent class** to that attribute value.
4. Compute the error rate for the attribute as the proportion of instances that do **not** belong to the majority class of their corresponding branch.

After evaluating all attributes, the attribute with the **smallest total error rate** is selected as the model.

### Prediction
Prediction is straightforward. For a new instance, the value of the selected attribute is checked, and the class assigned to the corresponding branch is returned.

### Handling Numeric Attributes
Numeric attributes are handled by **discretization**:
- Attribute values are sorted.
- Potential split points are considered where the class label changes.
- The split points are chosen to minimize the total classification error.

This discretization procedure is **highly sensitive to noise**. A single incorrectly labeled instance may introduce unnecessary intervals. A common countermeasure is to enforce a minimum number of instances in the majority class within each interval.

### Handling Missing Values
Missing values are treated as a **separate attribute value**. This allows the algorithm to process incomplete instances, but it may introduce additional branches.

### Strengths and Limitations
1R is extremely simple, fast, and interpretable. It often serves as a baseline classifier. However, it cannot model interactions between attributes and is sensitive to noise, especially when discretizing numeric attributes.

---

## 2.2 ID3 Decision Tree Learning Algorithm

### Basic Idea and Goal
ID3 constructs a **decision tree** using a greedy, top-down, divide-and-conquer strategy. Internal nodes test attributes, branches correspond to attribute values, and leaves assign class labels. The primary goal is to build a tree that is both **small** and **accurate**.

### Assumptions
The standard ID3 algorithm assumes:
- attributes are **nominal**,
- there are **no missing values**,
- the training data is relatively **noise-free**.

### Tree Construction Algorithm
The algorithm proceeds recursively as follows:
1. Select the best attribute for the current node.
2. Create one branch for each possible value of the selected attribute.
3. Split the training instances according to these branches.
4. Recursively apply the procedure to each subset of instances.

The recursion stops when:
- all instances at a node belong to the same class, or
- no further split improves purity.

### Entropy
Entropy is a measure of **impurity** or uncertainty in a set of instances. It is zero when all instances belong to the same class and maximal when classes are evenly distributed.

### Information Gain
To select the best attribute, ID3 uses **information gain**, which measures the expected reduction in entropy caused by splitting on an attribute:

Information Gain = Entropy(before split) − Entropy(after split)

The attribute with the highest information gain is chosen for the split.

### Properties of Entropy and Information Gain
A suitable impurity measure should:
- be zero if and only if the node is pure,
- be maximal when impurity is maximal,
- satisfy the multistage property.

Entropy is the only measure that satisfies all these properties.

### Highly Branching Attributes
Attributes with many distinct values (e.g. identification numbers) tend to produce many branches. This can artificially inflate information gain and lead to **overfitting**, because the resulting subsets are often pure by chance.

### Gain Ratio
To reduce the bias toward many-valued attributes, ID3 can be extended using the **gain ratio**. Gain ratio normalizes information gain by the intrinsic information of a split, which measures how much information is required to determine the branch an instance follows.

### Limitations of ID3
- Cannot handle numeric attributes
- Cannot handle missing values
- Sensitive to noise
- Prone to overfitting due to highly branching attributes

These limitations motivated the development of the C4.5 algorithm.

---

## 2.3 C4.5 Decision Tree Learning Algorithm

### Motivation
C4.5 is an extension of ID3 designed to handle **real-world data**, where noise, missing values, and numeric attributes are common.

### Handling Numeric Attributes
Numeric attributes are handled by creating **binary splits**. Candidate split points are evaluated, and the one maximizing gain ratio is selected. This allows C4.5 to work with continuous-valued data.

### Handling Missing Values
C4.5 handles missing values by assigning instances **fractionally** to all branches of a split, weighted according to the observed frequencies of attribute values. This allows learning and prediction without discarding incomplete instances.

### Attribute Selection Criterion
C4.5 uses **gain ratio** instead of pure information gain. This reduces the preference for attributes with many values and leads to more balanced trees.

### Pruning
Pruning is used to reduce overfitting and improve generalization.

#### Pre-pruning
Pre-pruning stops tree growth early if further splitting is unlikely to improve performance.

#### Post-pruning
Post-pruning first builds a full tree and then simplifies it.

### Pruning Operations
- **Subtree replacement**: replacing a subtree with a leaf
- **Subtree raising**: moving a subtree upward in the tree

Pruning decisions are based on estimates of classification error.

### Strengths and Limitations
C4.5 produces more robust and accurate trees than ID3 and is widely used in practice. However, it is computationally more expensive and still sensitive to noise in highly complex datasets.

---

### End of Topic 2

# 3. Association Rule Mining

---

## 3.1 Basic Idea and Goal

Association rule mining is a data mining task whose goal is to discover **regularities and co-occurrence relationships** between attribute values in large datasets. Unlike classification, there is **no distinguished class attribute**. Instead, the aim is to find rules that describe how the presence of certain items implies the presence of other items.

A typical informal interpretation is: *“If a set of events or items occurs together, then another event or item is also likely to occur.”*

This task is especially common in transactional data, such as shopping baskets, where each transaction contains a set of items.

---

## 3.2 Form of Association Rules

The standard form of an association rule is:

**IF X THEN Y**

where:
- **X** (left-hand side, antecedent) is a set of items,
- **Y** (right-hand side, consequent) is a set of items,
- X and Y are disjoint.

Association rules are similar to classification rules, but:
- there is no predefined class attribute,
- all attributes are treated symmetrically,
- attribute values are typically Boolean (present or not present).

---

## 3.3 Support and Confidence

To evaluate the quality of an association rule, two fundamental measures are used.

### Support

Support (also called coverage) measures how frequently the items in a rule occur together in the dataset.

For a rule **IF X THEN Y**, the support is the number (or proportion) of instances that contain **both X and Y**.

Support reflects how relevant or frequent a rule is in the data.

### Confidence

Confidence (also called accuracy) measures how often the rule is correct.

For a rule **IF X THEN Y**, confidence is defined as:

confidence = support(X ∧ Y) / support(X)

Confidence expresses the conditional probability that Y occurs given that X occurs.

---

## 3.4 The Challenge of Association Rule Mining

A naïve approach to association rule mining would be to treat every possible combination of attribute values as a separate class and apply rule learning. This approach is infeasible for two reasons:

1. **Computational complexity**: the number of possible rules grows exponentially with the number of attributes.
2. **Rule explosion**: the resulting number of rules would be extremely large and mostly uninteresting.

Therefore, association rule mining focuses on finding **only those rules that exceed predefined minimum thresholds** for support and confidence.

---

## 3.5 Items and Itemsets

- An **item** is a single attribute–value pair.
- An **itemset** is a set of items that occur together in a transaction.

The key idea is to first identify **frequent itemsets**, i.e. itemsets whose support is greater than or equal to a given minimum support threshold. Association rules are then generated from these frequent itemsets.

---

## 3.6 Frequent Itemsets and the Apriori Principle

The central observation behind efficient association rule mining is the **Apriori principle**:

> If an itemset is frequent, then all of its non-empty subsets must also be frequent.

Conversely, if an itemset is infrequent, none of its supersets can be frequent.

This property allows large parts of the search space to be pruned without explicitly evaluating them.

---

## 3.7 The Apriori Algorithm

The Apriori algorithm finds all frequent itemsets in a level-wise manner.

### Step-by-Step Description

1. **Find frequent 1-itemsets**  
   Scan the database and determine which individual items satisfy the minimum support threshold.

2. **Generate candidate k-itemsets**  
   Combine frequent (k−1)-itemsets to generate candidate k-itemsets.

3. **Prune candidates**  
   Remove candidate itemsets that contain any infrequent (k−1)-subset, using the Apriori principle.

4. **Count support**  
   Scan the database to determine the actual support of the remaining candidate itemsets.

5. **Repeat**  
   Discard infrequent itemsets and repeat the process for increasing values of k until no further frequent itemsets can be found.

This approach drastically reduces the number of itemsets whose support must be evaluated.

---

## 3.8 Generating Association Rules from Frequent Itemsets

Once all frequent itemsets have been found, association rules are generated from them.

For a frequent itemset X, all non-empty subsets A ⊂ X are considered. For each such subset, a rule of the form:

**IF A THEN (X − A)**

is generated, and its confidence is computed.

Only those rules whose confidence exceeds the minimum confidence threshold are retained.

The number of possible rules that can be generated from a frequent itemset of size N is 2^N − 1, which further highlights the importance of restricting rule generation to frequent itemsets.

---

## 3.9 Strengths and Limitations

### Strengths
- Simple and intuitive rule representation
- Unsupervised learning
- Widely applicable to transactional data

### Limitations
- Computationally expensive for large datasets
- Generates many redundant or uninteresting rules
- Requires careful choice of support and confidence thresholds

---

### End of Topic 3

# 4. Linear Regression, Numeric Prediction ("Regression")

---

## 4.1 Numeric Prediction as a Data Mining Task

Numeric prediction, often referred to as **regression**, is a supervised learning task where the goal is to predict a **numeric (continuous) target value** for unseen instances. Unlike classification, where the output is a nominal class label, regression produces real-valued outputs.

Typical examples include predicting prices, measurements, or quantities such as income, temperature, or demand.

---

## 4.2 Linear Regression: Basic Concept

Linear regression models the relationship between a numeric target variable and one or more numeric input attributes using a **linear function**.

The general form of a linear regression model is:

x̂ = w₀ + w₁a₁ + w₂a₂ + … + wₖaₖ

where:
- x̂ is the predicted numeric value,
- w₀ is the intercept (bias term),
- wᵢ are the weights (coefficients),
- aᵢ are the input attribute values.

The model assumes that the target variable can be approximated as a weighted sum of the input attributes.

---

## 4.3 Training Linear Regression Models

### Objective of Training

The goal of training is to find values for the weights such that the predictions are as close as possible to the true target values in the training data.

### Error Function

The most common error measure used during training is the **sum of squared errors (SSE)**:

SSE = ∑ (x⁽ⁱ⁾ − x̂⁽ⁱ⁾)²

where x⁽ⁱ⁾ is the true value and x̂⁽ⁱ⁾ is the predicted value for instance i.

Minimizing the squared error penalizes large errors more heavily than small ones.

### Weight Estimation

The optimal weights can be derived analytically using **matrix inversion** (normal equations). This approach requires that the number of training instances is sufficiently larger than the number of attributes. If this condition is not met, the solution may be unstable or undefined.

---

## 4.4 Prediction Phase

Once the weights have been learned, prediction for a new instance is performed by substituting the attribute values into the learned linear equation. The computation is fast and deterministic.

---

## 4.5 Linear Models for Classification

Linear models can also be used for **classification**, particularly in the binary case.

### Binary Classification

For two-class problems, the linear model defines a **decision boundary** (a line or hyperplane). The predicted class is determined by whether the output of the linear function is greater or less than a threshold (often zero).

### Multi-class Classification

For problems with more than two classes, multiple linear functions or higher-dimensional hyperplanes are used to separate the classes.

---

## 4.6 Logistic Regression

Logistic regression is a linear model adapted for classification tasks by transforming the linear output using a **sigmoid (logistic) function**.

### Logit Transformation

The sigmoid function maps real-valued inputs into the interval (0, 1), allowing the output to be interpreted as a **probability**:

P(class = 1 | a) = 1 / (1 + e^(−(w₀ + ∑ wᵢaᵢ)))

### Interpretation

Instead of predicting a numeric value directly, logistic regression models the probability that an instance belongs to a particular class. The final class decision is made by comparing this probability to a threshold.

### Training

The weights are learned using **maximum likelihood estimation**, rather than minimizing squared error.

---

## 4.7 Assumptions and Properties of Linear Regression

### Assumptions
- Linear relationship between attributes and target
- Numeric attributes
- Additive effects of attributes
- Limited multicollinearity

### Properties
- Simple and fast to train
- Highly interpretable model
- Sensitive to outliers
- Limited expressive power for nonlinear relationships

---

## 4.8 Strengths and Limitations

### Strengths
- Easy to interpret
- Efficient to compute
- Well-understood statistical properties

### Limitations
- Cannot model nonlinear patterns without transformation
- Sensitive to noise and outliers
- Requires careful handling of correlated attributes

---

### End of Topic 4

# 5. Support Vector Machine

---

## 5.1 Basic Idea and Motivation

Support Vector Machines (SVMs) are supervised learning methods primarily used for classification, but they can also be applied to numeric prediction (regression). The central idea of SVMs is to find a **decision boundary** that separates classes with the **maximum possible margin**. By maximizing this margin, SVMs aim to achieve good generalization performance on unseen data.

SVMs are especially effective in high-dimensional spaces and in cases where the number of attributes is large compared to the number of training instances.

---

## 5.2 Linear Classification and Hyperplanes

In the simplest case, SVMs are used for **binary classification** with linearly separable data.

A **hyperplane** is a linear decision boundary that separates the data into two classes. In a two-dimensional space, this boundary is a line; in higher dimensions, it is a plane or a higher-dimensional hyperplane.

Among all possible separating hyperplanes, SVMs choose the one that **maximizes the margin**, where the margin is defined as the distance between the hyperplane and the closest training instances from each class.

---

## 5.3 Support Vectors and Maximum Margin

The training instances that lie closest to the separating hyperplane are called **support vectors**. These instances are critical because they uniquely define the position of the optimal hyperplane.

Key properties:
- Only support vectors influence the final model
- Removing non-support-vector instances does not change the decision boundary
- The optimization problem has a unique, global optimum

Maximizing the margin leads to better robustness against noise and improved generalization.

---

## 5.4 Soft Margin and Non-Separable Data

In real-world data, perfect linear separability is rare. To handle overlapping classes, SVMs introduce the concept of a **soft margin**.

Soft margin SVMs allow some instances to violate the margin constraints by introducing **slack variables**. A regularization parameter controls the tradeoff between:
- maximizing the margin, and
- minimizing classification errors.

A larger regularization value penalizes misclassification more heavily, leading to narrower margins, while a smaller value allows more violations but yields wider margins.

---

## 5.5 Nonlinear Classification and Kernel Functions

Many datasets cannot be separated using a linear boundary in the original input space. SVMs address this problem using **kernel functions**.

The key idea is to implicitly map the data into a higher-dimensional feature space where a linear separation becomes possible. The kernel function computes inner products in this feature space without explicitly performing the transformation.

Common kernel functions include:
- **Polynomial kernel**: allows polynomial decision boundaries
- **Radial Basis Function (RBF) kernel**: creates flexible, nonlinear boundaries

The choice of kernel and its parameters strongly affects the performance of the SVM.

---

## 5.6 Optimization and Properties

Training an SVM corresponds to solving a **convex optimization problem**. Because the problem is convex, the solution is guaranteed to be globally optimal, unlike many other learning methods that may converge to local optima.

Important properties:
- Robust to overfitting in high-dimensional spaces
- Effective even when the number of features exceeds the number of instances
- Sensitive to parameter and kernel selection

---

## 5.7 Support Vector Regression (SVR)

Support Vector Machines can be adapted for numeric prediction using **Support Vector Regression**.

In SVR, the goal is to find a function that deviates from the true target values by at most a predefined margin (ε) for as many instances as possible. Deviations larger than ε are penalized.

SVR applies the same principles as classification SVMs, including margin maximization, slack variables, and kernel functions.

---

## 5.8 Strengths and Limitations

### Strengths
- Strong theoretical foundation
- Effective in high-dimensional spaces
- Robust generalization properties

### Limitations
- Computationally expensive for large datasets
- Sensitive to choice of kernel and parameters
- Less interpretable than decision trees or rules

---

### End of Topic 5

# 6. Bayesian Networks and Naive Bayes

---

## 6.1 Bayesian Reasoning and Bayes’ Theorem

Bayesian learning is based on **probabilistic reasoning under uncertainty**. The fundamental rule is **Bayes’ theorem**, which expresses how prior knowledge is updated using evidence:

P(H | E) = ( P(E | H) · P(H) ) / P(E)

where:
- H is a hypothesis (e.g. class label),
- E is the observed evidence (attribute values),
- P(H) is the prior probability of the hypothesis,
- P(E | H) is the likelihood of observing E if H is true.

Bayesian methods model uncertainty explicitly and allow reasoning even with incomplete information.

---

## 6.2 Naive Bayes Classifier

### Basic Idea

The **Naive Bayes (NB)** classifier is a probabilistic supervised learning algorithm. It predicts the most probable class for an instance based on all attribute values, assuming **conditional independence of attributes given the class**.

Although this assumption is almost never fully true in real data, Naive Bayes performs surprisingly well in practice.

---

### Knowledge Representation

Naive Bayes represents knowledge using:
- class prior probabilities P(C),
- conditional probability tables P(Aᵢ | C) for each attribute Aᵢ.

The model can also be interpreted as a **simple Bayesian network** where the class node is the parent of all attribute nodes.

---

### Learning Phase

During training, Naive Bayes estimates probabilities from the training data:
- Class priors are computed as relative class frequencies
- Conditional probabilities are computed as relative frequencies of attribute–class combinations

This corresponds to **Maximum Likelihood Estimation (MLE)**.

---

### Prediction Phase

For a new instance with attributes (A₁ = a₁, … , Aₙ = aₙ), the posterior probability of each class is computed as:

P(C | A₁,…,Aₙ) ∝ P(C) · ∏ P(Aᵢ = aᵢ | C)

The predicted class is the one with the highest posterior probability. Normalization is optional unless exact probabilities are required.

---

### Zero-Frequency Problem

If an attribute value never occurs with a particular class in the training data, the estimated probability becomes zero, eliminating that class entirely.

**Laplace smoothing** solves this problem by adding a small constant (usually 1) to all frequency counts, ensuring that no probability is zero.

---

### Handling Missing Values

- During training: missing values are ignored for probability estimation
- During prediction: attributes with missing values are omitted from the probability product

---

### Numeric Attributes

Numeric attributes are typically modeled using a **Gaussian (normal) distribution** per class.

For each numeric attribute and class, the mean and standard deviation are estimated from the training data and used to compute likelihoods.

---

### Properties of Naive Bayes

- Very fast training and prediction
- Handles irrelevant attributes well
- Sensitive to redundant or highly correlated attributes
- Widely used in text classification and spam filtering

---

## 6.3 Bayesian Networks

### Concept

A **Bayesian network** is a probabilistic graphical model representing a joint probability distribution using a **directed acyclic graph (DAG)**.

- Nodes represent random variables (attributes)
- Directed edges represent conditional dependencies
- Each node stores a conditional probability distribution given its parents

---

### Factorization of Joint Distribution

The joint probability distribution is factorized as:

P(A₁,…,Aₙ) = ∏ P(Aᵢ | Parents(Aᵢ))

This factorization makes complex distributions manageable and interpretable.

---

### Conditional Independence and d-Separation

Bayesian networks encode conditional independence assumptions.

Two nodes X and Y are **d-separated** by a set of nodes Z if all paths between X and Y are blocked according to d-separation rules. If d-separated, X and Y are conditionally independent given Z.

---

## 6.4 Prediction Using Bayesian Networks

Prediction involves computing class probabilities by:
1. Multiplying conditional probabilities along the network
2. Normalizing the resulting values

This generalizes Naive Bayes by allowing dependencies among attributes.

---

## 6.5 Learning Bayesian Networks

Learning consists of two tasks:
- **Parameter learning**: estimating conditional probabilities
- **Structure learning**: determining the graph structure

---

### Parameter Learning

If the network structure is known and data is complete, parameters are learned using **maximum likelihood estimation**.

---

### Structure Learning

If the structure is unknown, learning becomes a search problem over possible graphs:
- The quality of a network is evaluated using log-likelihood
- More edges increase likelihood but risk overfitting
- Complexity penalties or cross-validation are used

---

### K2 Algorithm

The **K2 algorithm** assumes a given ordering of variables and greedily adds parent edges that improve network score.

Properties:
- Efficient
- Sensitive to variable ordering
- Produces locally optimal structures

---

### Tree-Augmented Naive Bayes (TAN)

TAN extends Naive Bayes by allowing each attribute to have one additional parent besides the class node.

This relaxes the independence assumption while keeping inference efficient.

---

## 6.6 Strengths and Limitations

### Strengths
- Explicit uncertainty handling
- Clear probabilistic semantics
- Flexible model structure

### Limitations
- Structure learning is computationally expensive (NP-hard)
- Requires large amounts of data for reliable estimation
- Sensitive to incorrect independence assumptions

---

### End of Topic 6

# 7. Instance-Based Learning

---

## 7.1 Basic Idea and Motivation

Instance-based learning (also called **lazy learning** or **memory-based learning**) is a family of learning methods in which the training instances themselves represent the learned knowledge. Unlike eager learning algorithms (such as decision trees or Naive Bayes), instance-based learners do **not build an explicit model during training**. Instead, generalization is postponed until a new instance needs to be classified or predicted.

The simplest form of instance-based learning is **rote learning**, where all training instances are stored and reused directly.

---

## 7.2 Distance and Similarity Measures

The core concept of instance-based learning is **similarity**. A new instance is compared to stored instances using a distance (or similarity) measure.

### Numeric Attributes

For numeric attributes, the most commonly used distance measure is **Euclidean distance**. When multiple numeric attributes are present, distances are computed in a multidimensional space.

Because attributes may be measured on different scales, **normalization** is essential. Without normalization, attributes with large numeric ranges would dominate the distance computation.

To save computation time, the square root in Euclidean distance can be omitted because it does not affect relative distances (this is sometimes referred to as the city-block or squared-distance optimization).

### Nominal Attributes

For nominal attributes, distance is usually defined as:
- 0 if the values are equal
- 1 if the values are different

In some cases, nominal attributes may be weighted differently depending on their importance.

---

## 7.3 Types of Distance Linkage

When instance-based methods are extended to clustering or neighborhood definitions, different linkage strategies can be used:

- **Single linkage**: distance between the closest instances (sensitive to outliers)
- **Complete linkage**: distance between the farthest instances (also sensitive to outliers)
- **Average linkage**: average distance between all pairs (computationally expensive)
- **Centroid linkage**: distance between cluster centroids

---

## 7.4 k-Nearest Neighbors (k-NN)

### Basic Algorithm

The **k-nearest neighbors (k-NN)** algorithm is the most widely used instance-based learning method. It is a supervised learning algorithm primarily used for classification, but it can also be adapted for numeric prediction.

The algorithm works as follows:
1. Receive a new, unclassified instance
2. Compute its distance to all stored training instances
3. Select the k instances with the smallest distances
4. Determine the most frequent class among these neighbors
5. Assign this class to the new instance

---

### Choice of k

The parameter k controls the bias–variance tradeoff:
- Small k values are sensitive to noise but capture fine-grained structure
- Large k values smooth the decision boundary but may blur class distinctions

The optimal value of k is usually determined using **cross-validation**.

---

## 7.5 Problems and Limitations of k-NN

### Computational Cost

Prediction with k-NN is slow because distances must be computed to all stored instances. Storage requirements are also high because all instances must be kept.

### Noise Sensitivity

Noisy instances can significantly affect predictions, especially for small k values.

### Attribute Importance

All attributes are treated as equally important unless explicit weighting or attribute selection is applied.

### Interpretability

k-NN is often considered a **black-box method**, as it does not produce an explicit, human-interpretable model.

---

## 7.6 Improving Instance-Based Learning

Several strategies exist to address the limitations of basic k-NN.

### Instance Selection

Instead of storing all instances, only a subset is kept:
- **IB2**: incrementally stores only misclassified instances (sensitive to noise)
- **IB3**: monitors instance performance to remove noisy instances

### Attribute Selection and Weighting

Attribute selection can remove irrelevant features. Attribute weighting assigns greater importance to more relevant attributes. The **IB4** algorithm incorporates attribute weighting into distance computation.

### Rectangular Generalization

Rectangular generalization combines rule-based models inside regions of the feature space with nearest-neighbor methods outside those regions, improving interpretability and robustness.

---

## 7.7 Efficient Search with kd-Trees

### Motivation

To speed up nearest-neighbor search, specialized data structures can be used.

### kd-Tree Structure

A **kd-tree** recursively partitions the feature space into hyper-rectangles:
- The root represents the entire space
- Each split divides the space along one dimension
- Deeper nodes correspond to smaller regions

Splits are usually chosen along the dimension with the greatest variance, using the median as the split point.

---

### Querying and Updating

The efficiency of a kd-tree depends on its balance and depth. During search, backtracking may be required if nearby regions cannot be excluded.

Instance-based learning supports **incremental updates**. New instances can be inserted into a kd-tree by:
1. Finding the appropriate leaf
2. Inserting the instance
3. Splitting the leaf if necessary along the longest dimension

Periodic rebuilding of the tree may be required to maintain efficiency.

---

## 7.8 Strengths and Limitations

### Strengths
- Simple and intuitive
- No training phase
- Naturally supports incremental learning

### Limitations
- Slow prediction
- High memory usage
- Sensitive to noise and irrelevant attributes

---

### End of Topic 7

# 8. Clustering

---

## 8.1 Basic Idea and Motivation

Clustering is an **unsupervised learning** task whose goal is to divide a set of instances into groups called **clusters**, such that instances within the same cluster are similar to each other, while instances in different clusters are dissimilar. Unlike classification, clustering does not rely on predefined class labels.

Clustering is mainly a **knowledge representation and exploratory technique**. The meaning, number, and usefulness of clusters depend on the application and often require interpretation by a domain expert.

Typical applications include customer segmentation, image segmentation, document organization, and exploratory data analysis.

---

## 8.2 Clustering vs Classification

Although clustering and classification are different tasks, they are sometimes combined:

- **Clustering**: no labels, structure is inferred from data
- **Classification**: labels are given, structure is learned from examples

A hybrid approach is **semi-supervised learning**, where clustering is performed first and the cluster identifier is added as a new attribute to improve classification performance. This is useful when labeled data is scarce but unlabeled data is abundant.

---

## 8.3 k-Means Clustering

### Basic Idea

The **k-means** algorithm partitions the data into **k disjoint clusters**, where k is predefined by the user. Each cluster is represented by its **centroid**, which is the mean of the instances assigned to that cluster.

k-means produces flat (non-hierarchical), deterministic clusters.

---

### Algorithm

1. Choose k initial cluster centers (randomly or using heuristics)
2. Repeat until convergence:
   - Assign each instance to the nearest cluster center
   - Recompute cluster centers as the mean of assigned instances

The algorithm converges when assignments no longer change.

---

### Objective Function

k-means minimizes the **sum of squared distances** between instances and their assigned cluster centers. This objective is often called the **distortion function** or **total squared distance (TSD)**.

---

### Problems and Limitations

- Sensitive to the initial choice of cluster centers
- Requires predefined k
- Can converge to local minima
- Performs poorly on non-convex or non-isotropic clusters
- Sensitive to outliers

---

### Choosing the Number of Clusters

Several strategies exist:
- Minimizing TSD while penalizing larger k values
- Recursive k-means (splitting clusters with k = 2)
- Model selection criteria such as **Minimum Description Length (MDL)** or **Bayesian Information Criterion (BIC)**
- X-means algorithm (extends k-means using BIC)

---

### Practical Improvements

- Seed selection along directions of greatest variance
- Scaling for large datasets
- Mini-batch k-means for faster convergence
- Dimensionality reduction using PCA in high-dimensional spaces

---

## 8.4 Mean-Shift Clustering

The **mean-shift** algorithm is a centroid-based clustering method that identifies **dense regions** in the data.

### Key Characteristics

- Uses a sliding window that shifts toward regions of higher density
- Automatically determines the number of clusters
- Handles uneven cluster sizes well

### Limitations

- Computationally expensive
- Poor scalability for large datasets

---

## 8.5 Density-Based Clustering: DBSCAN

### Basic Idea

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** identifies clusters as regions of high density separated by regions of low density.

Two parameters define density:
- ε (epsilon): neighborhood radius
- minPts: minimum number of points required to form a dense region

---

### Algorithm Outline

1. Select an unvisited point
2. Retrieve all points within ε distance
3. If at least minPts neighbors exist, form a cluster
4. Expand the cluster by recursively adding density-reachable points
5. Points not belonging to any cluster are labeled as noise

---

### Properties

- Finds clusters of arbitrary shape
- Automatically determines the number of clusters
- Identifies outliers as noise

### Limitations

- Difficult to choose ε and minPts
- Struggles with clusters of varying density

---

## 8.6 Probability-Based Clustering

### Basic Idea

Probability-based clustering assigns instances to clusters **with a certain probability**, rather than making hard assignments.

Each cluster is modeled using a probability distribution, typically a **Gaussian distribution**.

---

### Gaussian Mixture Models (GMM)

A **Gaussian Mixture Model** represents the data as a mixture of a finite number of Gaussian components, each corresponding to a cluster.

---

### Expectation–Maximization (EM)

The **EM algorithm** is used to learn GMM parameters:

- **Expectation step**: compute the probability that each instance belongs to each cluster
- **Maximization step**: update distribution parameters based on these probabilities

The algorithm iterates until convergence to a (local) maximum of likelihood.

---

## 8.7 Hierarchical Clustering

Hierarchical clustering builds a hierarchy of clusters, represented as a **dendrogram**.

### Bottom-Up (Agglomerative) Clustering

1. Start with each instance as its own cluster
2. Repeatedly merge the two closest clusters
3. Continue until a single cluster remains

The sequence of merges defines the hierarchical structure.

---

### Top-Down (Divisive) Clustering

Starts with one large cluster and recursively splits it into smaller clusters.

---

### Distance Measures

Hierarchical clustering requires a distance or similarity measure between clusters, such as single linkage, complete linkage, average linkage, or centroid linkage.

---

## 8.8 Strengths and Limitations of Clustering

### Strengths
- No labeled data required
- Useful for exploration and pattern discovery
- Applicable to many domains

### Limitations
- Results depend on chosen parameters and distance measures
- Interpretation of clusters is subjective
- Evaluation is difficult without ground truth

---

### End of Topic 8

# 9. Data Transformations

---

## 9.1 Motivation and Role of Data Transformations

Data transformations are applied when the raw dataset is **not in an ideal form** for learning. Real-world data often violates the assumptions of learning algorithms (e.g. scale, distribution, missing values, noise). Appropriate transformations can significantly improve predictive performance, robustness, and computational efficiency.

Transformations must be **learned from the training data only** and then applied unchanged to validation and test data to avoid information leakage.

---

## 9.2 Simple Transformations and Sampling

### Time-Series Transformations

For time-dependent data, simple transformations can include:
- shifting values from past or future time steps
- computing differences between consecutive instances
- computing ratios of numeric attributes

If time steps are uneven, normalization is required. For frequency-domain analysis, **Fourier transformation** can be applied.

### Sampling

Sampling is useful when datasets are extremely large or when data arrives as a stream.

- **Random sampling**: select instances uniformly
- **Reservoir sampling**: maintain a fixed-size sample from a stream of unknown size

Sampling reduces computational cost while preserving statistical properties.

---

## 9.3 Attribute Scaling and Normalization

Many algorithms (e.g. k-means, SVM, PCA) assume:
- zero mean
- unit variance
- isotropic feature space

Scaling ensures that attributes measured on different scales contribute equally.

Common techniques include:
- min–max normalization
- standardization (zero mean, unit variance)

---

## 9.4 Attribute Selection and Modification

### Motivation

Irrelevant or redundant attributes can degrade performance, increase noise sensitivity, and slow down learning. Attribute selection often leads to:
- improved prediction accuracy
- faster training and prediction
- simpler and more interpretable models

Some algorithms (e.g. C4.5, instance-based learning) are especially sensitive to irrelevant attributes, while Naive Bayes is more robust.

---

### Filter Methods

Filter methods evaluate attributes **independently of the learning algorithm**.

Examples:
- removing attributes with many missing values
- low variance filtering
- correlation-based feature selection (CFS)

CFS selects subsets of attributes that are highly correlated with the class but weakly correlated with each other.

---

### Wrapper Methods

Wrapper methods evaluate attribute subsets **using a learning algorithm**.

- Performance is estimated using cross-validation or resubstitution error
- More accurate but computationally expensive
- Selected attributes depend on the learning scheme

Greedy search strategies include:
- forward selection
- backward elimination
- best-first and beam search

---

## 9.5 Principal Component Analysis (PCA)

### Basic Idea

**Principal Component Analysis (PCA)** is a linear, unsupervised dimensionality reduction technique. It transforms the data into a new coordinate system such that:
- the first component explains the maximum variance
- subsequent components explain decreasing variance
- components are orthogonal

---

### Algorithm

1. Standardize the data
2. Compute the covariance matrix
3. Find eigenvectors and eigenvalues
4. Select the top components
5. Project data into the new space

The scale of attributes affects PCA, so standardization is usually required.

---

### Properties and Variants

- PCA does not use class labels
- Preserves distance relationships approximately
- Incremental PCA exists but is computationally expensive
- Kernel PCA handles nonlinear structure

Related methods:
- **Linear Discriminant Analysis (LDA)**: supervised alternative maximizing class separation
- **Independent Component Analysis (ICA)**: finds statistically independent components
- Random projections can surprisingly preserve distances

---

## 9.6 Discretizing Numeric Attributes

### Motivation

Some algorithms require nominal attributes or perform better with discretized inputs. Numeric attributes also impose computational costs (e.g. sorting).

---

### Unsupervised Discretization

Does not use class labels:
- equal-width binning
- equal-frequency binning

Used when no class information is available (e.g. clustering).

---

### Supervised Discretization

Uses class labels to determine cut points:
- entropy-based methods
- C4.5-style binary splits

A common approach builds a decision tree on a single attribute using entropy as the splitting criterion and **Minimum Description Length (MDL)** as the stopping criterion.

---

### Encoding Discretized Attributes

A k-valued ordered attribute can be encoded as **k−1 binary attributes**, which preserves ordering better than integer coding.

---

## 9.7 Dirty Data and Noise Handling

### Sources of Dirty Data

- measurement errors
- typographical errors
- unsystematic noise
- duplicates and stale data

---

### Automatic Data Cleansing

Decision trees can be used to detect noise by:
- identifying misclassified instances
- removing them iteratively

This often produces smaller trees but only modest accuracy improvements.

---

### Outlier Detection and Robust Methods

- Visualization can reveal anomalies
- Ensemble agreement: remove instances misclassified by multiple learners

For numeric prediction, **robust regression** minimizes absolute error rather than squared error and may remove extreme outliers.

---

## 9.8 Transforming Multi-Class Problems into Binary Problems

Some learning algorithms are inherently binary.

### Strategies

- **One-vs-rest**: one classifier per class
- **One-vs-one**: classifier for each class pair

---

### Error-Correcting Output Codes (ECOC)

ECOC assigns codewords to classes with large Hamming distances. This allows:
- correction of single-bit errors
- improved robustness

Limitations:
- difficult to achieve row and column separation for small class counts
- works best when number of classes > 3

---

## 9.9 Probability Calibration

Accurate probability estimates are important for cost-sensitive decisions.

Predicted probabilities may be:
- overly optimistic (close to 0 or 1)
- overly pessimistic (close to uniform)

Calibration improves probability quality without changing classification accuracy.

---

### End of Topic 9

# 10. Ensemble Learning

---

## 10.1 Motivation: Bias–Variance Tradeoff

The goal of supervised learning is to build models that generalize well to unseen data. Two fundamental sources of error are:

- **Bias**: error caused by overly simple models that cannot capture important patterns (underfitting)
- **Variance**: error caused by overly complex models that fit noise in the training data (overfitting)

Ensemble learning aims to reduce generalization error by **combining multiple models**. The key idea is that while individual models may make errors, their errors may cancel out when aggregated.

---

## 10.2 Basic Idea of Ensemble Learning

An ensemble consists of multiple **base learners** (also called experts). Each base learner is trained to solve the same task, and their predictions are combined using a voting or averaging scheme.

- For **classification**, predictions are combined by majority or weighted voting
- For **numeric prediction**, predictions are combined by averaging

Ensembles are often treated as black-box models, as interpretability is reduced compared to single models.

---

## 10.3 Stacking

### Concept

**Stacking** (stacked generalization) combines different learning algorithms trained on the same dataset.

- **Level-0 (base level)**: multiple diverse learning algorithms (e.g. decision trees, Bayesian networks)
- **Level-1 (meta level)**: a meta-learner that combines the outputs of base learners

---

### Training Procedure

To avoid overfitting:
- Base learners are trained using cross-validation
- Their predictions on held-out folds are used as input features for the meta-learner
- The meta-learner is trained only on these predictions

The meta-learner is often simple, as most learning is done at level 0.

---

## 10.4 Bagging (Bootstrap Aggregating)

### Basic Idea

**Bagging** builds multiple models of the same type, each trained on a different dataset generated by **sampling with replacement** from the original training set.

Each sampled dataset has the same size as the original dataset, but contains duplicate instances.

---

### Prediction

- Classification: majority voting
- Numeric prediction: averaging

Bagging is particularly effective for **unstable learning algorithms**, where small changes in training data lead to large changes in the model (e.g. decision trees).

---

### Properties

- Reduces variance
- Does not significantly increase bias
- Often improves performance on noisy data

Bagging with costs uses averaging instead of voting and provides better probability estimates.

---

## 10.5 Randomization

Instead of randomizing the input data, **randomization** introduces randomness directly into the learning algorithm.

Examples:
- random attribute selection in decision trees
- random subsets of instances in k-NN
- random initial weights in neural networks

Randomization increases diversity among base learners and can be combined with bagging.

---

## 10.6 Random Forests

### Concept

A **random forest** is an ensemble of decision trees built using both bagging and randomization.

At each split:
- Only a random subset of attributes is considered

This increases diversity among trees while keeping individual trees reasonably accurate.

---

### Properties

- Strong predictive performance
- Robust to noise
- Reduced overfitting compared to single trees
- Less interpretable than a single decision tree

---

## 10.7 Boosting

### Basic Idea

**Boosting** builds ensembles incrementally. Each new model focuses on instances that were misclassified by previous models.

Boosting works well with **weak learners**, i.e. models that perform only slightly better than random guessing.

---

### AdaBoost

AdaBoost adjusts weights of training instances:
- Misclassified instances receive higher weights
- Correctly classified instances receive lower weights

Predictions are combined using weighted voting.

Theoretical guarantees show that training error decreases exponentially if each base learner performs better than random.

---

## 10.8 Gradient Boosting

Gradient boosting views boosting as an **optimization problem** over function space.

- Each new model approximates the negative gradient of a loss function
- Allows custom loss functions

---

### XGBoost

XGBoost is an optimized implementation of gradient boosting with:
- regularization to reduce overfitting
- efficient handling of sparse data
- built-in cross-validation
- support for parallel processing

XGBoost often outperforms deep learning on structured tabular data.

---

## 10.9 Strengths and Limitations

### Strengths
- Improved predictive accuracy
- Robustness to noise
- Flexible combination of models

### Limitations
- Reduced interpretability
- Increased computational cost
- More complex model selection

---

### End of Topic 10

# 11. Logistic Regression, Perceptron, Winnow, Neural Networks

---

## 11.1 Overview of Linear and Nonlinear Models

This topic groups together several learning algorithms that are historically and conceptually related. Logistic regression, perceptron, and Winnow are **linear models**, meaning that their decision boundaries are linear in the input space. Neural networks generalize these models by combining multiple linear units with nonlinear activation functions, allowing the representation of complex, nonlinear relationships.

---

## 11.2 Logistic Regression

### Basic Idea

Logistic regression is a **probabilistic linear classifier**. Instead of predicting a class label directly, it models the **probability** that an instance belongs to a given class.

The model computes a linear combination of input attributes:

z = w₀ + w₁a₁ + w₂a₂ + … + wₖaₖ

and transforms it using the **logistic (sigmoid) function**:

P(class = 1 | a) = 1 / (1 + e^(−z))

This maps arbitrary real-valued inputs into the interval (0, 1).

---

### Interpretation

Logistic regression models the **log-odds** (logit) of the class probability as a linear function of the input attributes. This allows outputs to be interpreted as probabilities, which is important for cost-sensitive decisions.

---

### Training

The parameters are learned using **maximum likelihood estimation**. Unlike linear regression, logistic regression does not minimize squared error, because squared error leads to poor probability estimates.

Optimization is typically performed using gradient-based methods.

---

### Properties

- Linear decision boundary
- Probabilistic output
- Robust to irrelevant attributes
- Requires numeric or encoded nominal attributes

---

## 11.3 Perceptron

### Basic Idea

The perceptron is one of the earliest learning algorithms for binary classification. It computes a weighted sum of input attributes and assigns a class based on the sign of the result.

---

### Learning Algorithm

The perceptron is a **mistake-driven algorithm**:

1. Initialize all weights to zero or small random values
2. For each training instance:
   - Predict the class
   - If the prediction is wrong, update the weights

The weight update increases weights for attributes that support the correct class and decreases weights for those that support the incorrect class.

---

### Convergence Properties

The perceptron converges to a correct solution **if and only if** the data is linearly separable. If the data is not linearly separable, the algorithm will never converge.

---

### Limitations

- No probabilistic output
- Sensitive to noise
- Cannot learn non-linear decision boundaries

---

## 11.4 Winnow Algorithm

### Motivation

Winnow is designed for problems with a **large number of attributes**, where only a small subset is relevant.

---

### Learning Mechanism

Winnow uses **multiplicative weight updates** instead of additive updates:

- Weights are multiplied by a factor greater than 1 for correct attributes
- Weights are divided by a factor for incorrect attributes

This allows Winnow to quickly focus on relevant attributes and ignore irrelevant ones.

---

### Properties

- Works well in high-dimensional spaces
- Robust to many irrelevant attributes
- Requires careful choice of update parameters

---

## 11.5 Neural Networks

### Basic Structure

A neural network consists of layers of interconnected units called **neurons**.

- **Input layer**: receives attribute values
- **Hidden layers**: perform nonlinear transformations
- **Output layer**: produces predictions

Each neuron computes a weighted sum of its inputs followed by a nonlinear activation function.

---

### Activation Functions

Common activation functions include:
- sigmoid
- hyperbolic tangent
- rectified linear unit (ReLU)

Nonlinear activation functions are essential for learning complex patterns.

---

### Learning and Backpropagation

Neural networks are trained using **backpropagation**, which applies gradient descent to minimize a loss function.

The algorithm alternates between:
- forward propagation of inputs
- backward propagation of errors

---

### Properties

- Can model highly nonlinear relationships
- Require large amounts of data
- Sensitive to parameter settings
- Often difficult to interpret

---

## 11.6 Comparison of Methods

- Logistic regression: probabilistic, interpretable, linear
- Perceptron: simple, mistake-driven, requires separability
- Winnow: effective in high-dimensional sparse spaces
- Neural networks: powerful but complex and opaque

---

### End of Topic 11

# 12. Performance Evaluation – Measures

---

## 12.1 Why Performance Evaluation Is Needed

The goal of data mining and machine learning is not only to learn a model from data, but to **estimate how well the model will perform on unseen data**. A model that fits the training data perfectly may still perform poorly on new data due to overfitting.

Performance evaluation provides objective criteria for:
- comparing different learning algorithms,
- selecting model parameters,
- estimating generalization ability,
- understanding tradeoffs between different types of errors.

---

## 12.2 Confusion Matrix for Classification

For classification tasks, performance is often summarized using a **confusion matrix**.

For a binary classification problem:

- **True Positives (TP)**: positive instances correctly classified
- **False Positives (FP)**: negative instances incorrectly classified as positive
- **True Negatives (TN)**: negative instances correctly classified
- **False Negatives (FN)**: positive instances incorrectly classified as negative

The confusion matrix is the basis for most classification performance measures.

---

## 12.3 Accuracy and Error Rate

### Accuracy

Accuracy is the proportion of correctly classified instances:

accuracy = (TP + TN) / (TP + TN + FP + FN)

### Error Rate

The error rate is the proportion of incorrectly classified instances:

error = 1 − accuracy

Accuracy is easy to compute but may be misleading for **imbalanced datasets**, where one class dominates.

---

## 12.4 Precision, Recall, and F-Measure

### Precision

Precision measures how many instances predicted as positive are actually positive:

precision = TP / (TP + FP)

High precision means few false positives.

---

### Recall (Sensitivity)

Recall measures how many actual positive instances are correctly identified:

recall = TP / (TP + FN)

High recall means few false negatives.

---

### F-Measure

The **F-measure** combines precision and recall into a single value using the harmonic mean:

F = 2 · (precision · recall) / (precision + recall)

The harmonic mean penalizes extreme values and emphasizes balance between precision and recall.

---

## 12.5 Tradeoffs Between Precision and Recall

Increasing precision often decreases recall, and vice versa. The appropriate tradeoff depends on the application:

- Medical diagnosis: high recall is crucial
- Spam filtering: high precision is often preferred

The decision threshold of a classifier controls this tradeoff.

---

## 12.6 Cost-Sensitive Evaluation

In many applications, different types of errors have **different costs**.

- False negatives may be much more expensive than false positives (or vice versa)
- Accuracy alone does not capture these differences

Cost-sensitive evaluation assigns explicit costs to each type of error and aims to minimize **expected cost** rather than error rate.

---

## 12.7 ROC Curves

### Basic Idea

A **Receiver Operating Characteristic (ROC) curve** visualizes classifier performance across different decision thresholds.

- x-axis: false positive rate (FP / (FP + TN))
- y-axis: true positive rate (recall)

Each point corresponds to a different threshold.

---

### Interpretation

- A classifier close to the diagonal behaves like random guessing
- A curve closer to the top-left corner indicates better performance

---

### Area Under the Curve (AUC)

The **AUC** summarizes ROC performance as a single value:
- AUC = 1: perfect classifier
- AUC = 0.5: random classifier

AUC is insensitive to class distribution and is useful for comparing classifiers.

---

## 12.8 Lift Charts

Lift charts are commonly used in marketing and decision-support applications.

They show how much better a classifier performs compared to random selection when instances are sorted by predicted probability.

---

## 12.9 Kappa Statistic

The **Kappa statistic** measures agreement between predicted and true class labels while accounting for agreement by chance.

- Kappa = 1: perfect agreement
- Kappa = 0: agreement equivalent to chance

Kappa is especially useful when class distributions are skewed.

---

## 12.10 Evaluation of Numeric Prediction

For numeric prediction (regression), classification measures are not applicable.

### Mean Squared Error (MSE)

MSE = (1/n) · ∑ (x − x̂)²

Large errors are penalized strongly.

---

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and has the same units as the target variable.

---

### Mean Absolute Error (MAE)

MAE = (1/n) · ∑ |x − x̂|

MAE is more robust to outliers than MSE.

---

## 12.11 Choosing Appropriate Measures

The choice of performance measure depends on:
- task type (classification vs regression)
- class distribution
- cost of errors
- application domain

No single measure is universally optimal.

---

### End of Topic 12

# 13. Performance Evaluation – Datasets

---

## 13.1 Goal of Dataset-Based Evaluation

Performance measures define *how* quality is quantified, but **dataset-based evaluation methods** define *how performance is estimated*. Because the true generalization error on unseen data is unknown, it must be estimated using available data in a careful and principled way.

A poor evaluation strategy can lead to overly optimistic or pessimistic estimates and incorrect model comparisons.

---

## 13.2 Training, Validation, and Test Sets

In supervised learning, data is commonly divided into disjoint subsets:

- **Training set**: used to learn model parameters
- **Validation set**: used for model selection and parameter tuning
- **Test set**: used only once to estimate final performance

The test set must remain untouched during learning to avoid biased performance estimates.

---

## 13.3 Resubstitution Error

### Definition

The **resubstitution error** is obtained by evaluating a model on the **same data used for training**.

### Properties

- Computationally cheap
- Strongly **optimistic** estimate of performance
- Severely affected by overfitting

Resubstitution error is generally unsuitable for realistic performance estimation but can be useful for debugging.

---

## 13.4 Holdout Method

### Basic Idea

The **holdout method** splits the dataset into two disjoint parts:

- training set
- test set

A typical split ratio is 2/3 for training and 1/3 for testing.

---

### Properties

- Simple and fast
- Performance estimate depends strongly on the particular split
- Not reliable for small datasets

Repeated holdout can reduce variance but increases computational cost.

---

## 13.5 Cross-Validation

### k-Fold Cross-Validation

In **k-fold cross-validation**, the dataset is divided into k equally sized folds:

1. One fold is used as the test set
2. The remaining k−1 folds form the training set
3. The process is repeated k times

The final performance estimate is the average over all k runs.

---

### Stratified Cross-Validation

In **stratified k-fold cross-validation**, each fold preserves the original class distribution. This is especially important for imbalanced datasets.

A common choice is **10-fold stratified cross-validation**, which offers a good tradeoff between bias and variance.

---

### Leave-One-Out Cross-Validation (LOO)

Leave-one-out cross-validation is an extreme case of k-fold cross-validation where k equals the number of instances.

Properties:
- Almost unbiased
- Very high variance
- Computationally expensive

---

## 13.6 Bootstrap Methods

### Basic Idea

**Bootstrap** evaluation repeatedly samples datasets of the same size as the original dataset **with replacement**.

Each bootstrap sample contains:
- about 63.2% unique instances
- the remaining instances are duplicates

Instances not selected form the test set.

---

### 0.632 Bootstrap

The **0.632 bootstrap** combines:
- training error on the bootstrap sample
- test error on the out-of-bag instances

This corrects for the optimistic bias of resubstitution error.

---

## 13.7 Comparing Learning Algorithms

To compare algorithms fairly:
- use the **same dataset splits** for all algorithms
- apply identical evaluation protocols
- perform statistical tests if differences are small

Common issues include multiple testing and dependence between folds.

---

## 13.8 Choosing an Evaluation Method

The choice of evaluation method depends on:
- dataset size
- computational cost
- variance–bias tradeoff

Guidelines:
- Large datasets: holdout or repeated holdout
- Medium datasets: stratified k-fold cross-validation
- Small datasets: cross-validation or bootstrap

---

## 13.9 Summary of Dataset-Based Evaluation Methods

- Resubstitution: fast, optimistic
- Holdout: simple, high variance
- Cross-validation: balanced and widely used
- Bootstrap: robust but computationally expensive

---

### End of Topic 13
