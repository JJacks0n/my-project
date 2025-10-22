# Predictive Provenance System - Analysis Report

## Data Statistics
- **Total Sequences**: 1801
- **Average Sequence Length**: 5.7
- **Behavior Types**: 4

### Behavior Distribution
- **systematic**: 1233 (68.5%)
- **analyst**: 477 (26.5%)
- **focused**: 90 (5.0%)
- **explorer**: 1 (0.1%)

## Model Performance
- **Random Forest**: 0.981
- **Gradient Boosting**: 0.992

## Top Interaction Patterns
1. **('select', 'select', 'select')**: 3573 occurrences
2. **('select', 'navigate', 'select')**: 1138 occurrences
3. **('navigate', 'select', 'select')**: 1061 occurrences
4. **('select', 'select', 'navigate')**: 936 occurrences
5. **('assist', 'select', 'select')**: 463 occurrences
6. **('select', 'assist', 'select')**: 418 occurrences
7. **('select', 'select', 'assist')**: 354 occurrences
8. **('select', 'select', 'deselect')**: 279 occurrences
9. **('select', 'deselect', 'select')**: 194 occurrences
10. **('select', 'select', 'modify_selection')**: 154 occurrences

## Export Information
- **Export Time**: 2025-10-21 17:38:20
- **Data Source**: projection-space-explorer repository
- **Models Exported**: 2 traditional models + 1 neural model
