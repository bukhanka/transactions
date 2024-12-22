# Payment System Analysis Results

```python
{'payments': {'basic_stats': {'total_transactions': 393032, 'unique_cards': 92965, 'currencies': ['UZS', 'RUB', 'AZN', 'TRY', 'KZT', 'BRL', 'EUR', 'PHP', 'JPY', 'AUD', 'MYR', 'USD', 'GHS', 'MXN', 'PEN', 'KRW', 'THB', 'HKD'], 'total_volume_usd': np.float64(8473892.272403331), 'avg_transaction_usd': np.float64(21.56031130392266)}, 'currency_stats':      amount                                       amount_usd                 cardToken
      count           sum       mean         std         sum    mean     std   nunique
cur                                                                                   
AUD      26  1.022930e+03      39.34       51.04      664.91   25.57   33.18        20
AZN   33237  5.414611e+05      16.29       41.87   317764.54    9.56   24.57      9732
BRL   56413  5.313416e+06      94.19      411.23   915942.53   16.24   70.89         1
EUR    1618  1.040219e+05      64.29      127.96   108373.00   66.98  133.31       176
GHS       2  1.659000e+02      82.95       31.18       11.28    5.64    2.12         1
HKD       1  7.070000e+01      70.70         NaN        9.08    9.08     NaN         1
JPY     407  5.285030e+04     129.85      183.29      341.47    0.84    1.18         1
KRW       1  1.040000e+02     104.00         NaN        0.07    0.07     NaN         1
KZT   10892  8.380567e+07    7694.24    26967.89   167862.75   15.41   54.02      4039
MXN       3  5.575000e+02     185.83       12.87       27.29    9.10    0.63         1
MYR      12  9.537000e+02      79.48       76.83      213.45   17.79   17.19         1
PEN       1  2.400000e+02     240.00         NaN       63.31   63.31     NaN         1
PHP       4  7.872000e+02     196.80       93.72       13.36    3.34    1.59         1
RUB  150724  5.197195e+08    3448.15     6159.98  4982550.44   33.06   59.06     21199
THB       2  3.297200e+03    1648.60     1836.50       95.63   47.81   53.26         1
TRY    4872  5.157053e+06    1058.51     2752.30   149250.27   30.63   79.65       610
USD       3  2.590380e+03     863.46      108.38     2590.38  863.46  108.38         2
UZS  134814  2.343742e+10  173850.02  1321782.61  1828118.52   13.56  103.10     57192, 'card_stats': {'total_cards': 92965, 'multi_currency_cards': 2, 'avg_transactions_per_card': np.float64(4.227741623191524), 'max_transactions_per_card': np.int64(152680)}}, 'providers': {'provider_stats': {'day1_providers': 511, 'day2_providers': 426, 'currencies_supported': ['UZS', 'RUB', 'AZN', 'TRY', 'KZT', 'BRL', 'EUR', 'PHP', 'JPY', 'AUD', 'MYR', 'USD', 'GHS', 'MXN', 'PEN', 'KRW', 'THB', 'HKD']}, 'provider_changes': {np.int64(0): {'conversion_change': np.float64(0.19999999999999996), 'commission_change': np.float64(0.0), 'time_change': np.float64(-4.0)}, np.int64(1): {'conversion_change': np.float64(-0.09999999999999998), 'commission_change': np.float64(0.0049999999999999975), 'time_change': np.float64(18.0)}, np.int64(2): {'conversion_change': np.float64(0.15000000000000002), 'commission_change': np.float64(0.010000000000000002), 'time_change': np.float64(10.0)}, np.int64(3): {'conversion_change': np.float64(0.19999999999999996), 'commission_change': np.float64(0.0), 'time_change': np.float64(-2.0)}, np.int64(4): {'conversion_change': np.float64(0.0), 'commission_change': np.float64(-0.0050000000000000044), 'time_change': np.float64(-18.0)}, np.int64(5): {'conversion_change': np.float64(0.15000000000000002), 'commission_change': np.float64(0.0050000000000000044), 'time_change': np.float64(8.0)}, np.int64(6): {'conversion_change': np.float64(0.1499999999999999), 'commission_change': np.float64(0.03), 'time_change': np.float64(4.0)}, np.int64(7): {'conversion_change': np.float64(-0.1499999999999999), 'commission_change': np.float64(0.0), 'time_change': np.float64(-8.0)}, np.int64(8): {'conversion_change': np.float64(-0.050000000000000044), 'commission_change': np.float64(0.010000000000000002), 'time_change': np.float64(-2.0)}, np.int64(9): {'conversion_change': np.float64(-0.050000000000000044), 'commission_change': np.float64(0.0049999999999999975), 'time_change': np.float64(0.0)}, np.int64(10): {'conversion_change': np.float64(0.050000000000000044), 'commission_change': np.float64(0.023000000000000003), 'time_change': np.float64(6.0)}, np.int64(11): {'conversion_change': np.float64(0.19999999999999996), 'commission_change': np.float64(-0.010000000000000002), 'time_change': np.float64(-4.0)}, np.int64(12): {'conversion_change': np.float64(0.09999999999999998), 'commission_change': np.float64(0.0), 'time_change': np.float64(12.0)}, np.int64(13): {'conversion_change': np.float64(0.15000000000000002), 'commission_change': np.float64(-0.002999999999999999), 'time_change': np.float64(4.0)}, np.int64(14): {'conversion_change': np.float64(-0.15000000000000002), 'commission_change': np.float64(0.035), 'time_change': np.float64(-4.0)}, np.int64(15): {'conversion_change': np.float64(-0.04999999999999993), 'commission_change': np.float64(-0.015), 'time_change': np.float64(0.0)}, np.int64(16): {'conversion_change': np.float64(0.09999999999999998), 'commission_change': np.float64(-0.015), 'time_change': np.float64(-12.0)}, np.int64(17): {'conversion_change': np.float64(0.1499999999999999), 'commission_change': np.float64(0.015), 'time_change': np.float64(2.0)}, np.int64(18): {'conversion_change': np.float64(0.19999999999999996), 'commission_change': np.float64(0.0), 'time_change': np.float64(-2.0)}, np.int64(19): {'conversion_change': np.float64(0.050000000000000044), 'commission_change': np.float64(-0.009999999999999995), 'time_change': np.float64(-4.0)}, np.int64(20): {'conversion_change': np.float64(-0.19999999999999996), 'commission_change': np.float64(-0.023000000000000003), 'time_change': np.float64(4.0)}, np.int64(21): {'conversion_change': np.float64(0.09999999999999998), 'commission_change': np.float64(0.0), 'time_change': np.float64(12.0)}, np.int64(22): {'conversion_change': np.float64(-0.19999999999999996), 'commission_change': np.float64(-0.0049999999999999975), 'time_change': np.float64(-10.0)}, np.int64(23): {'conversion_change': np.float64(-0.050000000000000044), 'commission_change': np.float64(0.015), 'time_change': np.float64(-16.0)}, np.int64(24): {'conversion_change': np.float64(0.09999999999999998), 'commission_change': np.float64(0.002999999999999999), 'time_change': np.float64(-6.0)}, np.int64(25): {'conversion_change': np.float64(0.09999999999999998), 'commission_change': np.float64(0.0), 'time_change': np.float64(18.0)}, np.int64(26): {'conversion_change': np.float64(0.0), 'commission_change': np.float64(-0.015), 'time_change': np.float64(-18.0)}, np.int64(27): {'conversion_change': np.float64(-0.25), 'commission_change': np.float64(0.0), 'time_change': np.float64(4.0)}, np.int64(28): {'conversion_change': np.float64(-0.15000000000000002), 'commission_change': np.float64(-0.018), 'time_change': np.float64(4.0)}, np.int64(29): {'conversion_change': np.float64(-0.15000000000000002), 'commission_change': np.float64(0.0050000000000000044), 'time_change': np.float64(2.0)}, np.int64(30): {'conversion_change': np.float64(0.09999999999999998), 'commission_change': np.float64(-0.010000000000000002), 'time_change': np.float64(2.0)}, np.int64(31): {'conversion_change': np.float64(0.0), 'commission_change': np.float64(-0.0050000000000000044), 'time_change': np.float64(-4.0)}, np.int64(32): {'conversion_change': np.float64(0.0), 'commission_change': np.float64(0.0), 'time_change': np.float64(2.0)}, np.int64(33): {'conversion_change': np.float64(0.19999999999999996), 'commission_change': np.float64(-0.018), 'time_change': np.float64(8.0)}, np.int64(34): {'conversion_change': np.float64(-0.04999999999999993), 'commission_change': np.float64(0.0049999999999999975), 'time_change': np.float64(14.0)}, np.int64(35): {'conversion_change': np.float64(0.050000000000000044), 'commission_change': np.float64(-0.03), 'time_change': np.float64(-10.0)}, np.int64(36): {'conversion_change': np.float64(0.0), 'commission_change': np.float64(-0.0049999999999999975), 'time_change': np.float64(12.0)}, np.int64(37): {'conversion_change': np.float64(0.1499999999999999), 'commission_change': np.float64(-0.013000000000000001), 'time_change': np.float64(-2.0)}, np.int64(38): {'conversion_change': np.float64(0.09999999999999998), 'commission_change': np.float64(-0.0049999999999999975), 'time_change': np.float64(-8.0)}, np.int64(39): {'conversion_change': np.float64(0.0), 'commission_change': np.float64(-0.010000000000000002), 'time_change': np.float64(-2.0)}, np.int64(40): {'conversion_change': np.float64(0.0), 'commission_change': np.float64(-0.015), 'time_change': np.float64(-12.0)}}, 'currency_support':             ID CONVERSION                     ... LIMIT_MIN              LIMIT_MAX                       
         count       mean     std  min   max  ...       min      max          mean        min         max
CURRENCY                                      ...                                                        
AUD         32     0.6266  0.0729  0.5  0.75  ...    1000.0  91000.0  2.071562e+07  1000000.0  36900000.0
AZN         92     0.6141  0.0865  0.5  0.75  ...    1000.0  91000.0  1.970870e+07  1000000.0  39900000.0
BRL         54     0.6231  0.0904  0.5  0.75  ...    1000.0  91000.0  2.121111e+07  2100000.0  39400000.0
EUR         51     0.6157  0.0815  0.5  0.75  ...    1000.0  91000.0  2.142941e+07  1800000.0  39200000.0
GHS         79     0.6323  0.0760  0.5  0.75  ...    1000.0  91000.0  2.200886e+07  1000000.0  39800000.0
HKD         36     0.6153  0.0852  0.5  0.75  ...    1000.0  91000.0  2.226667e+07  2300000.0  39300000.0
JPY         73     0.6185  0.0922  0.5  0.75  ...    1000.0  91000.0  1.726712e+07  1400000.0  39400000.0
KRW         14     0.6179  0.0912  0.5  0.75  ...    1000.0  91000.0  1.690714e+07  4400000.0  39900000.0
KZT         56     0.6045  0.0844  0.5  0.75  ...    1000.0  91000.0  2.266250e+07  2800000.0  39000000.0
MXN         39     0.6436  0.0940  0.5  0.75  ...    1000.0  91000.0  1.775385e+07  1000000.0  38900000.0
MYR         46     0.6522  0.0925  0.5  0.75  ...    1000.0  91000.0  1.911957e+07  1300000.0  39600000.0
PEN         49     0.6388  0.0825  0.5  0.75  ...    1000.0  91000.0  2.116327e+07  1900000.0  37700000.0
PHP         39     0.6423  0.0815  0.5  0.75  ...    1000.0  91000.0  2.051282e+07  1500000.0  39800000.0
RUB         92     0.6228  0.0833  0.5  0.75  ...    1000.0  91000.0  2.156522e+07  1700000.0  39600000.0
THB         41     0.6280  0.0844  0.5  0.75  ...    1000.0  91000.0  1.772195e+07  4800000.0  39300000.0
TRY         55     0.6273  0.0907  0.5  0.75  ...    1000.0  91000.0  1.946000e+07  1900000.0  38700000.0
USD         53     0.6085  0.0903  0.5  0.75  ...    1000.0  91000.0  1.896415e+07  1000000.0  39300000.0
UZS         36     0.6250  0.0975  0.5  0.75  ...    1000.0  91000.0  1.746944e+07  2100000.0  36400000.0

[18 rows x 19 columns]}, 'volume': {'daily_volumes':                         amount         amount_usd        
                           sum  count         sum    mean
cur eventTimeRes                                         
AUD 2024-11-24    4.320000e+01      3       28.08    9.36
    2024-11-25    2.282900e+02     10      148.39   14.84
    2024-11-26    7.514400e+02     13      488.44   37.57
AZN 2024-11-24    2.946835e+04   1898    17293.94    9.11
    2024-11-25    2.618580e+05  15793   153675.29    9.73
    2024-11-26    2.501347e+05  15546   146795.31    9.44
BRL 2024-11-24    1.222810e+05   1316    21079.17   16.02
    2024-11-25    3.480617e+06  39592   599999.25   15.15
    2024-11-26    1.710517e+06  15505   294864.11   19.02
EUR 2024-11-24    8.667040e+03    113     9029.57   79.91
    2024-11-25    4.774652e+04    795    49743.71   62.57
    2024-11-26    4.760832e+04    710    49599.72   69.86
GHS 2024-11-25    6.090000e+01      1        4.14    4.14
    2024-11-26    1.050000e+02      1        7.14    7.14
HKD 2024-11-26    7.070000e+01      1        9.08    9.08
JPY 2024-11-24    1.273600e+03      7        8.23    1.18
    2024-11-25    3.273020e+04    235      211.47    0.90
    2024-11-26    1.884650e+04    165      121.77    0.74
KRW 2024-11-26    1.040000e+02      1        0.07    0.07
KZT 2024-11-24    4.359693e+06    338     8732.47   25.84
    2024-11-25    4.771926e+07   6083    95581.68   15.71
    2024-11-26    3.172671e+07   4471    63548.61   14.21
MXN 2024-11-25    5.575000e+02      3       27.29    9.10
MYR 2024-11-25    9.537000e+02     12      213.45   17.79
PEN 2024-11-25    2.400000e+02      1       63.31   63.31
PHP 2024-11-25    4.974000e+02      3        8.44    2.81
    2024-11-26    2.898000e+02      1        4.92    4.92
RUB 2024-11-24    3.579172e+07   6766   343135.26   50.71
    2024-11-25    2.291429e+08  65842  2196793.03   33.36
    2024-11-26    2.547848e+08  78116  2442622.15   31.27
THB 2024-11-26    3.297200e+03      2       95.63   47.81
TRY 2024-11-24    3.319388e+05    370     9606.64   25.96
    2024-11-25    2.950276e+06   2414    85383.93   35.37
    2024-11-26    1.874839e+06   2088    54259.71   25.99
USD 2024-11-25    2.590380e+03      3     2590.38  863.46
UZS 2024-11-24    5.760224e+08   2692    44929.75   16.69
    2024-11-25    1.130535e+10  63341   881817.36   13.92
    2024-11-26    1.155604e+10  68781   901371.41   13.10, 'limit_analysis':     currency  provider_id  avg_daily_volume  limit_min_usd  limit_max_usd  meets_min  within_max  potential_penalty
0        UZS            0         609372.84          4.758         1006.2       True       False            0.00000
1        UZS            1         609372.84          1.638         1404.0       True       False            0.00000
2        UZS            1         609372.84          6.318          842.4       True       False            0.00000
3        UZS            0         609372.84          7.098         1294.8       True       False            0.00000
4        UZS            1         609372.84          1.638         1185.6       True       False            0.00000
..       ...          ...               ...            ...            ...        ...         ...                ...
932      HKD           30              9.08       1413.071      2903218.6      False        True           14.13071
933      HKD           32              9.08      10405.341      2749065.4      False        True          104.05341
934      HKD           32              9.08       2697.681      1669993.0      False        True           26.97681
935      HKD           29              9.08       6551.511      2170990.9      False        True           65.51511
936      HKD           29              9.08       9120.731      4881518.0      False        True           91.20731

[937 rows x 8 columns]}}
```