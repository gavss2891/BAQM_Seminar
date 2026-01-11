# Null Value Handling Documentation

## Overview
This document explains how null values were handled in the dataset. The analysis identified several columns with missing data, and different strategies were applied based on the column type and missing data patterns.

## Missing Data Analysis

### Initial Assessment
The analysis revealed the following columns with missing values:
- **holidayEventName**: ~96% missing
- **discountPct**: ~87% missing  
- **promo_id**: ~87% missing
- **FSC_index**: ~0.005% missing (18 rows)

## Handling Strategy

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

### 2. Promotion ID (promo_id)

**Handling**: No imputation was performed. The `promo_id` column was kept as-is since:
- Missing values indicate no promotion (which is valid information)
- A binary flag `has_promo` was created for analysis purposes

### 3. Holiday Event Name (holidayEventName)

**Handling**: No imputation was performed. This column was not used in the analysis due to the extremely high missing rate (~96%).

### 4. FSC Index (FSC_index)

**Status**: **NOT HANDLED** - Only visualized

The FSC_index had very few missing values (18 rows, ~0.005%). Visualizations were created to explore the missing data patterns, but no imputation strategy was implemented at this stage.

## Key Points

- Only `discountPct` nulls were actively imputed
- Imputation used historical data from the same article to maintain data integrity
- Mode (most frequent value) was chosen over mean/median to preserve typical discount patterns
- Missing values in other columns were either left as-is or excluded from analysis
- FSC_index missing values require further investigation before handling
