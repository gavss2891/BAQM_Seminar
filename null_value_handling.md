# Null Value Handling Documentation

### 1. Discount Percentage (discountPct)

**Problem**: Many rows had a `promo_id` but missing `discountPct` values.

**Solution**: Implemented a multi-tiered imputation strategy using historical discount data:

1. **Primary Method**: For each missing discount, looked up past discount percentages for the same article:
   - First preference: Mode discount from the same month in previous years
   - Second preference: Overall mode discount from all past dates for that article

2. **Fallback Method**: If no past discounts were found for the same article:
   - Used the mode discount from another promotion for the same article (closest date)

3. **Implementation**: 
   - Created a lookup table (`past_discounts_missing.csv`) with historical discount statistics
   - Applied the imputation only to rows where `promo_id` exists but `discountPct` is null
   - Used mode (most frequent value) as the imputation method to maintain consistency with typical discount patterns

### 2. FSC Index (FSC_index)
**Problem**: 18 rows missing `FSC_index` values for ONE article.

**Status**: **NOT HANDLED** - Only visualized

The FSC_index had very few missing values (18 rows, ~0.005%). Visualizations were created to explore the missing data patterns, but no imputation strategy was implemented at this stage. c

I do not know a good trategy to handle this. 

It does not seem to be much of a concern as non of the missing values are in december + very few missing values relative to all transactions. 
