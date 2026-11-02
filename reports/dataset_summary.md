# Attesta dataset summary

## Class counts (ecosystem x split)

|                   |   benign |   malicious |
|:------------------|---------:|------------:|
| ('npm', 'test')   |     1823 |        1160 |
| ('npm', 'train')  |    14601 |        9258 |
| ('npm', 'val')    |     1823 |        1160 |
| ('pypi', 'test')  |      764 |         489 |
| ('pypi', 'train') |     6114 |        3912 |
| ('pypi', 'val')   |      764 |         490 |

## Aux label rates by class

|                       |   aux_install_exec |   aux_network |   aux_cred_access |   aux_obfuscation |   aux_process_spawn |
|:----------------------|-------------------:|--------------:|------------------:|------------------:|--------------------:|
| ('npm', 'benign')     |              0.023 |         0.150 |             0.043 |             0.004 |               0.083 |
| ('npm', 'malicious')  |              0.680 |         0.442 |             0.133 |             0.135 |               0.320 |
| ('pypi', 'benign')    |              0.071 |         0.332 |             0.131 |             0.005 |               0.216 |
| ('pypi', 'malicious') |              0.522 |         0.671 |             0.130 |             0.138 |               0.414 |

## Parse success (AST vs regex fallback, per file)

|                               |        ast |     regex |   ast_success |
|:------------------------------|-----------:|----------:|--------------:|
| ('npm', 'aux_cred_access')    | 123169.000 | 36280.000 |         0.772 |
| ('npm', 'aux_install_exec')   |  29786.000 |    39.000 |         0.999 |
| ('npm', 'aux_network')        | 123169.000 | 36280.000 |         0.772 |
| ('npm', 'aux_obfuscation')    | 123169.000 | 36280.000 |         0.772 |
| ('npm', 'aux_process_spawn')  | 123169.000 | 36280.000 |         0.772 |
| ('pypi', 'aux_cred_access')   | 112880.000 |  2916.000 |         0.975 |
| ('pypi', 'aux_install_exec')  |   7411.000 |   214.000 |         0.972 |
| ('pypi', 'aux_network')       | 112880.000 |  2916.000 |         0.975 |
| ('pypi', 'aux_obfuscation')   | 112880.000 |  2916.000 |         0.975 |
| ('pypi', 'aux_process_spawn') | 112880.000 |  2916.000 |         0.975 |

## Size skew (bytes)

|             |    50% |    mean |       max |
|:------------|-------:|--------:|----------:|
| ('npm', 0)  |  38548 | 1225227 | 139783324 |
| ('npm', 1)  |   6994 | 1972041 | 316321913 |
| ('pypi', 0) | 115500 | 2649591 | 169965660 |
| ('pypi', 1) |   5459 |  616732 | 173192853 |

## Popularity bucket share by class

|             |   low |   mid |   top |   unknown |
|:------------|------:|------:|------:|----------:|
| ('npm', 0)  | 0.616 | 0.161 | 0.220 |     0.003 |
| ('npm', 1)  | 0.000 | 0.000 | 0.000 |     1.000 |
| ('pypi', 0) | 0.565 | 0.309 | 0.125 |     0.000 |
| ('pypi', 1) | 0.000 | 0.000 | 0.000 |     1.000 |

## Compromised-package group (report separately)

|             |   n |
|:------------|----:|
| ('npm', 0)  |  50 |
| ('npm', 1)  | 941 |
| ('pypi', 1) |  19 |

## Sanity checks

- Shortcut AUC (train->test): size+popularity **1.000**, size only 0.796, popularity only 1.000

- total_chars only (near-empty-filter shortcut check): **0.776**

- Cluster overlap across splits: {'test&train': 0, 'test&val': 0, 'train&val': 0}; package overlap: {'test&train': 0, 'test&val': 0, 'train&val': 0}

- Exact-hash label conflicts: 1; dropped by cluster cap: 14237

## Spot check (3 malicious + 3 benign)

|   label | eco   | pkg                      | ver    |   install_exec |   network |   cred_access |   obfuscation |   process_spawn | evidence                  |
|--------:|:------|:-------------------------|:-------|---------------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
|       1 | pypi  | trove4j                  | 0.1.1  |              0 |         1 |             0 |             0 |               0 | aux_network:import socket |
|       1 | npm   | stardrop-windows-x64-bas | 1.1.44 |              0 |         0 |             0 |             0 |               0 | -                         |
|       1 | npm   | synqroomkit              | 0.1.0  |              0 |         0 |             0 |             0 |               0 | -                         |
|       0 | npm   | lazycrate                | 1.0.4  |              0 |         0 |             0 |             0 |               0 | -                         |
|       0 | pypi  | colorprintconsoled       | 0.1.1  |              0 |         0 |             0 |             0 |               0 | -                         |
|       0 | pypi  | rfc3986-validator        | 0.1.1  |              0 |         0 |             0 |             0 |               0 | -                         |
