# Day 2 Case Study: "The Feature That Doubled Accuracy"

## Context
A SaaS company built a **customer churn prediction model** to reduce monthly customer loss ($5M revenue impact). Initial accuracy was only **72%**—not good enough for business use.

## The Problem

### Initial Model (72% Accuracy)
The team used raw features directly from the database:
- Customer age
- Account balance
- Total login count
- Total support tickets
- Days since account creation

**Result**: Model couldn't distinguish churners from loyal customers. Too many false positives.

### Root Cause: Missing Domain Knowledge
- Engineers focused on what was **easy to extract** (raw database fields)
- Data scientists didn't collaborate with **customer success team** (who know churn patterns)
- No feature engineering (transformation of raw features into meaningful signals)
- Model had no idea what makes a customer likely to churn

## The Breakthrough: Collaboration + Feature Engineering

### Interview with Customer Success Team
**Question**: "What patterns do you see in customers before they churn?"

**Answers**:
- Customers who suddenly stop using the product (recency matters!)
- Customers with high support ticket volume but low purchases (trouble, not loyalty!)
- Customers whose activity suddenly changes (anomaly detection)
- Support escalations (technical problems = churn risk)

### Engineered Features (12 new features)

**Recency Features** (when was the customer last active?):
```
days_since_last_login = max(date) - last_login_date
days_since_last_purchase = max(date) - last_purchase_date
login_frequency_30d = count(logins in last 30 days)
```

**Frequency Features** (how often do they engage?):
```
support_tickets_per_month = total_support_tickets / months_active
purchases_per_month = total_purchases / months_active
login_velocity = (current_month_logins - prior_month_logins)
```

**Interaction Features** (relationships between features):
```
support_intensity = support_tickets / purchase_amount
  # High support + low spend = customers in trouble
  
support_to_login_ratio = support_tickets / login_count
  # Logging in just to complain? = bad signal
  
purchase_consistency = std(purchase_amounts_per_month)
  # Erratic spending = churn risk?
```

**Derived Features** (business logic):
```
at_risk_flag = (days_since_last_login > 30) AND (support_tickets_per_month > 2)
  # Haven't logged in 30 days + complaining? = leaving soon

customer_health_score = weighted combination of features
  # Single score summarizing customer engagement
```

### The Results

**Model Accuracy Evolution**:
- Baseline (raw features only): **72%**
- After adding recency features: **78%** (+6 points)
- After adding frequency features: **82%** (+4 points)
- After adding interaction features: **87%** (**+15 points total**)

**Business Impact**:
- Precision (when we predict churn, how often right?): 89%
- Recall (catching actual churners): 84%
- Ability to **intervene early**: 6 weeks advance notice (time for sales to retain customer)

### Validation

The team tested feature importance (which features matter most?):
```
Feature Importance (Random Forest):
1. support_intensity (22%)           ← Strongest signal!
2. days_since_last_login (18%)
3. support_to_login_ratio (15%)
4. purchase_consistency (12%)
5. login_velocity (10%)
... and 7 others
```

**Key insight**: Support intensity was 22x more predictive than raw "total support tickets" because it's **contextualized by purchase behavior**.

---

## Why Feature Engineering Mattered

### Principle: Good Features >> Complex Models

```
Bad (72% acc): Raw data + sophisticated model
└─ Model is smart, but data is dumb

Good (87% acc): Engineered features + simple model
└─ Data tells a story, model just has to listen

Better (89% acc): Engineered features + good model
└─ Both data and model are optimized
```

### The 80/20 Rule in ML Performance
- **80% of model performance** comes from **good features**
- **20% of model performance** comes from **model choice**

**Implication**: Spend more time on feature engineering, less time tuning hyperparameters!

---
## Key Lesson 

> **"80% of ML performance comes from good features, not complex models. Domain knowledge + feature engineering beats fancy algorithms."**

### How to Engineer Features Like This

**Step 1: Understand the Problem Domain**
- Talk to business people (customer success, product, sales)
- What do they already know about customer behavior?
- What patterns do they manually look for?

**Step 2: Translate Domain Knowledge → Features**
- Recency: When did customer last do something?
- Frequency: How often do they engage?
- Interaction: How do different behaviors relate?
- Derived: What business logic can we encode?

**Step 3: Validate with Data**
- Does the feature correlate with target (churn)?
- Is there a clear pattern?
- Can you explain the relationship?

**Step 4: Iterate**
- Build baseline with raw features
- Add engineered features incrementally
- Measure impact of each feature
- Keep what works, remove noise

---

## Common Feature Engineering Mistakes (Avoid These)

❌ **Only using raw database fields** ("We can only log what's easy to extract")
- Solution: Engineering takes work, but pays off massively

❌ **Creating too many features** ("More features = better model?")
- Solution: Start with 3-5 high-impact features, add incrementally

❌ **Not validating feature logic** ("This seemed like a good idea")
- Solution: Plot feature vs. target; does the relationship make sense?

❌ **Ignoring domain expertise** ("I'm a data scientist; I don't need business input")
- Solution: Collaborate! Business people know patterns you'll never discover in data

---

## Enterprise Context: Feature Stores

At scale, companies use **Feature Stores** (centralized feature repositories):
- Store engineered features, not raw data
- Version features (just like code versioning)
- Reuse features across multiple models
- Track lineage (feature A comes from raw data B)
- Monitor feature quality (detect when feature generation breaks)

Example: Uber's **Michelangelo** or Netflix's **Metaflow** manage thousands of engineered features for dozens of models.

**But for now**: Master basic feature engineering. Feature stores are infrastructure for scale.

---

## Discussion Questions for 

1. **Domain Knowledge**: What domain do you work in? What "obvious" patterns do domain experts see that data might not surface?

2. **Feature Brainstorming**: If you were building a recommendation system, what features would predict "user engagement"?
   - Recency: days since last click
   - Frequency: clicks per week
   - Interaction: diversity of recommendations clicked
   - What else?

3. **Validation**: How would you verify that a feature is actually useful (beyond just model accuracy)?
