# Global oasis water-support change

## Research purpose

Core calculations for *Global oasis water-support change under equal-oasis and fixed-area weighting*. The analysis compares 3,437 fixed oases between 1987–2005 and 2006–2024, with complementary distribution and fixed-size-quintile summaries for 1984–2021. Water support is the water-class quantity divided by valid spatial support, on a 0–1 scale.

## Code

- `period_change.py`: equal-oasis and fixed-area period changes for the global set and five continents, shared whole-trajectory reweighting ranges, and continental contributions to the global area-weighted change.
- `distribution.py`: global annual means, quantiles and Gini coefficients, whole-trajectory trend resampling, and an original-valid-observation diagnostic.
- `size_gradient.py`: fixed global area quintiles, annual area-weighted means, trends and paired Q5−Q1 contrasts, globally and within continents.
- `analysis_input.py` and `size_kernel.py`: shared input handling and numerical kernels.

Each calculation writes numerical CSV files to its output directory. Period-change and size outputs are on the proportion scale unless a column explicitly ends in `_pp`; multiply proportion differences by 100 for percentage points. Distribution trend columns ending in `_pp_per_year` already include that conversion. For `gini_slope`, the value is the change in 100 × Gini per year, not percentage points of a water-support proportion; the output column name is retained to match the numerical implementation.

The period-change ranges use 2,000 shared resamples of 3,437 complete trajectories with the original group count or area denominator held fixed. They are central empirical reweighting ranges conditional on the input values and fixed set, not conventional within-group mean confidence intervals. Regional contributions use the global total-area denominator. Distribution and size resampling follow their separate estimands; they do not resample spatial blocks or propagate the complete measurement process.

## Dependencies

Use Python 3.13 and install the two numerical dependencies:

```bash
python -m pip install -r requirements.txt
```

## Inputs and execution

CSV and gzip-compressed CSV inputs are accepted. Read identifiers as strings; `NA` is a valid geographic code.

| Input | Required fields and content |
|---|---|
| Annual analysis, 1987–2024 | `OasisID`, `calendar_year`, `water_support_ratio`; 130,606 unique keys, 3,437 objects × 38 complete years, proportions in 0–1. |
| Membership | `source_oasis_id`, `corrected_oasis_id`, `fixed_2020_area_km2`, `continent5`; one row per fixed object. Annual identifiers must match either the complete source namespace or the complete corrected namespace. Continent codes are `AF`, `AS`, `NA`, `OA`, `SA`. |
| Annual baseline, 1984–2021 | `OasisID`, `calendar_year`, `water_support_ratio`, `repair_class`; the same 3,437 objects × 38 complete years. `ORIGINAL_VALID` identifies records used in the original-only diagnostic. Other values remain in the complete analysis. |
| Fixed area attributes | `oasis_id`, `area_km2`, `continent`; one row per baseline object, using its matching identifier namespace and fixed reference area. Continent labels are `Africa`, `Asia`, `North America`, `Oceania`, `South America`. |

For the period calculation, `analysis_value`, if supplied, takes priority over `water_support_ratio`; subsequent fallbacks are `water_ratio`, `final_available_water_ratio`, and `source_final_available_water_ratio`. For distribution and size calculations, `water_support_ratio` takes priority over `final_available_water_ratio`. Use the intended complete analysis column rather than substituting an incomplete legacy column.

```bash
python period_change.py --inputs annual_1987_2024.csv.gz --membership membership.csv --output results/periods --workers 4
python distribution.py --water annual_1984_2021.csv.gz --area fixed_area.csv --output results/distribution
python size_gradient.py --water annual_1984_2021.csv.gz --area fixed_area.csv --output results/size --workers 4
```

The period calculation produces six area-weighted summaries, six equal-oasis summaries and five continental contributions. The global distribution calculation produces 76 annual rows across complete and original-only branches and five distribution-trend estimates. The size calculation produces 1,140 annual rows, 30 trend rows, six Q5−Q1 trend contrasts and 228 annual contrasts. Area ties are ordered by stable object identifier before assigning fixed global quintiles; groups are not redefined each year or within each continent.

## Data access and references

Execution requires the study-specific complete annual analysis tables, fixed-oasis membership and area metadata described in the manuscript. Their preparation includes retrospective model filling and object-identity reconciliation; obtaining the satellite products alone does not supply these analysis inputs. Access to the corresponding derived inputs must be arranged with the study authors.

The historical source is JRC Global Surface Water v1.4 YearlyHistory for 1984–2021. The 2022–2024 analysis tail derives annual water-class support from v1.5 MonthlyHistory_2022_2024: valid monthly classes determine valid support, and seasonal plus permanent annual water classes form the numerator. This annual derivation is distinct from monthly water frequency. The historical segment retains v1.4; later JRC corrections for 2016–2021 have not been substituted into these input definitions. Product access and version information are available from the [JRC Global Surface Water download service](https://global-surface-water.appspot.com/download).

Pekel, J.-F., Cottam, A., Gorelick, N., & Belward, A. S. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature*, 540, 418–422. https://doi.org/10.1038/nature20584

Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical Society: Series B*, 69, 369–390. https://doi.org/10.1111/j.1467-9868.2007.00593.x

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57, 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
