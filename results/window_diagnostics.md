# Window diagnostics v2 (auto-generated)

Lead time N=5; kept windows only.

## local channel by split × anchor
|                           |   windows |   events |   unk |   nonempty |   unk_rate |
|:--------------------------|----------:|---------:|------:|-----------:|-----------:|
| ('test', 'hard')          |      1883 |    25497 | 22523 |      1     |     0.8834 |
| ('test', 'pos')           |      1085 |     1878 |   545 |      0.138 |     0.2902 |
| ('test', 'uniform')       |     14346 |    55926 | 35866 |      0.127 |     0.6413 |
| ('train', 'hard')         |      3337 |    65123 |     0 |      1     |     0      |
| ('train', 'pos')          |      2239 |    19878 |     0 |      0.162 |     0      |
| ('train', 'uniform')      |     36801 |   303560 |     0 |      0.089 |     0      |
| ('validation', 'hard')    |      1405 |    11720 |     0 |      1     |     0      |
| ('validation', 'pos')     |       647 |      564 |     0 |      0.134 |     0      |
| ('validation', 'uniform') |     13740 |    65214 | 12942 |      0.134 |     0.1985 |

## global channel by split × anchor
|                           |   windows |           events |              unk |   nonempty |   unk_rate |
|:--------------------------|----------:|-----------------:|-----------------:|-----------:|-----------:|
| ('test', 'hard')          |      1883 |      2.08569e+06 |      1.69817e+06 |      1     |     0.8142 |
| ('test', 'pos')           |      1085 | 263201           | 146647           |      0.466 |     0.5572 |
| ('test', 'uniform')       |     14346 |      7.2188e+06  |      4.46234e+06 |      0.637 |     0.6182 |
| ('train', 'hard')         |      3337 |      6.43217e+06 |      0           |      1     |     0      |
| ('train', 'pos')          |      2239 |      1.41753e+06 |      0           |      0.608 |     0      |
| ('train', 'uniform')      |     36801 |      4.66025e+07 |      0           |      0.682 |     0      |
| ('validation', 'hard')    |      1405 |      1.04097e+06 |      0           |      1     |     0      |
| ('validation', 'pos')     |       647 | 300135           |   5162           |      0.669 |     0.0172 |
| ('validation', 'uniform') |     13740 |      8.02773e+06 | 253124           |      0.75  |     0.0315 |

## train / local: top texts in POSITIVE windows (664)
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| ciod: cpu <*> at treeaddr <*> sent unrecognized message <*>                                                    | train        |    2165 |
| ciod: Message code <*> is not <*> or <*>                                                                       | train        |    2026 |
| generating core.<*>                                                                                            | train        |     616 |
| ciod: Error loading <*>: invalid or missing program image, No such file or directory                           | train        |     616 |
| Node card is not fully functional                                                                              | train        |     305 |
| Can not get assembly information for node card                                                                 | train        |     305 |
| Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserte | train        |     304 |
| ciod: LOGIN chdir(<*>) failed: No such file or directory                                                       | train        |     232 |
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |      97 |
| ciod: pollControlDescriptors: Detected the debugger died.                                                      | train        |      69 |
| Lustre mount FAILED : <*> : point <*>                                                                          | train        |      56 |
| ciod: Y coordinate <*> exceeds physical dimension <*> at line <*> of node map file <*>                         | train        |      48 |
| ciod: Error creating node map from file <*>: Permission denied                                                 | train        |      48 |
| Lustre mount FAILED : <*> : block_id : location                                                                | train        |      40 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>                                      | train        |      31 |

## train / local: top texts in NEGATIVE windows (41713)
| masked text                                                                                                  | first seen   |   lines |
|:-------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| generating core.<*>                                                                                          | train        |  173164 |
| <*> double-hummer alignment exceptions                                                                       | train        |   31725 |
| <*> floating point alignment exceptions                                                                      | train        |   24370 |
| ciod: Error loading <*>: invalid or missing program image, Exec format error                                 | train        |   23951 |
| Lustre mount FAILED : <*> : point <*>                                                                        | train        |   21062 |
| instruction cache parity error corrected                                                                     | train        |   11162 |
| CE sym <*>, at <*>, mask <*>                                                                                 | train        |    9693 |
| ciod: LOGIN chdir(<*>) failed: No such file or directory                                                     | train        |    6806 |
| ciod: failed to read message prefix on control stream (CioStream socket to <*>:<*>                           | train        |    6591 |
| ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to <*>:<*>: Link has been severed  | train        |    5973 |
| iar <*> dear <*>                                                                                             | train        |    5859 |
| ciod: Error loading <*>: invalid or missing program image, No such file or directory                         | train        |    3690 |
| ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to <*>:<*>: Link has been severed | train        |    3241 |
| ciod: generated <*> core files for program <*>                                                               | train        |    2420 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>                                    | train        |    2412 |

## train / global: top texts in POSITIVE windows (664)
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| idoproxydb hit ASSERT condition: ASSERT expression=<*> Source file=<*> Source line=<*> Function=int IdoTranspo | train        |  193256 |
| generating core.<*>                                                                                            | train        |  167397 |
| ciod: Error loading <*>: invalid or missing program image, No such file or directory                           | train        |   32123 |
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |   28775 |
| instruction cache parity error corrected                                                                       | train        |   24822 |
| ciod: LOGIN chdir(<*>) failed: No such file or directory                                                       | train        |   24695 |
| Lustre mount FAILED : <*> : point <*>                                                                          | train        |   15907 |
| Ido packet timeout                                                                                             | train        |    9666 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>                                      | train        |    4645 |
| total of <*> ddr error(s) detected and corrected                                                               | train        |    4289 |
| Can not get assembly information for node card                                                                 | train        |    3622 |
| Node card is not fully functional                                                                              | train        |    3608 |
| Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserte | train        |    3502 |
| ciod: failed to read message prefix on control stream (CioStream socket to <*>:<*>                             | train        |    3048 |
| ciod: cpu <*> at treeaddr <*> sent unrecognized message <*>                                                    | train        |    2165 |

## train / global: top texts in NEGATIVE windows (41713)
| masked text                                                                                                    | first seen   |    lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|---------:|
| generating core.<*>                                                                                            | train        | 23916643 |
| <*> double-hummer alignment exceptions                                                                         | train        |  3228174 |
| ciod: Error loading <*>: invalid or missing program image, Exec format error                                   | train        |  3098553 |
| <*> floating point alignment exceptions                                                                        | train        |  2284914 |
| data TLB error interrupt                                                                                       | train        |  2214366 |
| Lustre mount FAILED : <*> : point <*>                                                                          | train        |  2064042 |
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |  1446160 |
| instruction cache parity error corrected                                                                       | train        |   801162 |
| iar <*> dear <*>                                                                                               | train        |   693287 |
| ciod: LOGIN chdir(<*>) failed: No such file or directory                                                       | train        |   569124 |
| ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to <*>:<*>: Link has been severed    | train        |   508959 |
| ciod: failed to read message prefix on control stream (CioStream socket to <*>:<*>                             | train        |   402604 |
| ciod: generated <*> core files for program <*>                                                                 | train        |   363695 |
| idoproxydb hit ASSERT condition: ASSERT expression=<*> Source file=<*> Source line=<*> Function=int IdoTranspo | train        |   358263 |
| instruction address: <*>                                                                                       | train        |   350182 |

## validation / local: top texts in POSITIVE windows (193)
| masked text                                                                          | first seen   |   lines |
|:-------------------------------------------------------------------------------------|:-------------|--------:|
| CE sym <*>, at <*>, mask <*>                                                         | train        |      71 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>            | train        |      55 |
| ciod: Error loading <*>: invalid or missing program image, No such file or directory | train        |      26 |
| total of <*> ddr error(s) detected and corrected                                     | train        |      25 |
| <*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected                  | train        |       6 |
| <*> torus receiver y+ input pipe error(s) (dcr <*>) detected and corrected           | train        |       5 |
| <*> torus receiver x+ input pipe error(s) (dcr <*>) detected and corrected           | train        |       5 |
| ddr: activating redundant bit steering: rank=<*> symbol=<*>                          | train        |       4 |
| <*> torus receiver z+ input pipe error(s) (dcr <*>) detected and corrected           | train        |       4 |
| <*> L3 EDRAM error(s) (dcr <*>) detected and corrected                               | train        |       3 |
| <*> torus receiver x- input pipe error(s) (dcr <*>) detected and corrected           | train        |       3 |
| <*> torus receiver y- input pipe error(s) (dcr <*>) detected and corrected           | train        |       3 |
| <*> torus receiver z- input pipe error(s) (dcr <*>) detected and corrected           | train        |       2 |

## validation / local: top texts in NEGATIVE windows (15599)
| masked text                                                                                                  | first seen   |   lines |
|:-------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| iar <*> dear <*>                                                                                             | train        |   15211 |
| data cache search parity error detected. attempting to correct                                               | validation   |   12863 |
| CE sym <*>, at <*>, mask <*>                                                                                 | train        |    7706 |
| ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to <*>:<*>: Link has been severed  | train        |    5604 |
| ciod: Error loading <*>: invalid or missing program image, Permission denied                                 | train        |    3400 |
| <*> floating point alignment exceptions                                                                      | train        |    3062 |
| Lustre mount FAILED : <*> : block_id : location                                                              | train        |    2219 |
| ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to <*>:<*>: Link has been severed | train        |    1948 |
|                                                                                                              | train        |    1732 |
| ciod: Error loading <*>: invalid or missing program image, No such file or directory                         | train        |    1592 |
| ciod: generated <*> core files for program <*>                                                               | train        |    1562 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>                                    | train        |    1429 |
| total of <*> ddr error(s) detected and corrected                                                             | train        |    1162 |
| instruction cache parity error corrected                                                                     | train        |     680 |
| suppressing further interrupts of same type                                                                  | train        |     622 |

## validation / local: top UNSEEN (UNK) texts in POSITIVE windows
| masked text   | first seen   | lines   |
|---------------|--------------|---------|

## validation / global: top texts in POSITIVE windows (193)
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |   10978 |
| iar <*> dear <*>                                                                                               | train        |    8547 |
| ciod: Error loading <*>: invalid or missing program image, No such file or directory                           | train        |    2840 |
| instruction cache parity error corrected                                                                       | train        |    2637 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>                                      | train        |    2228 |
| total of <*> ddr error(s) detected and corrected                                                               | train        |    1629 |
|                                                                                                                | train        |    1522 |
| rts: kernel terminated for reason <*>                                                                          | train        |    1024 |
| Error receiving packet on tree network, expecting type <*> instead of type <*> (softheader=<*> <*> <*> <*>) PS | validation   |    1024 |
| <*> floating point alignment exceptions                                                                        | train        |    1024 |
| <*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected                                            | train        |     644 |
| ciod: Error loading <*>: invalid or missing program image, Permission denied                                   | train        |     512 |
| <*> L3 EDRAM error(s) (dcr <*>) detected and corrected                                                         | train        |     348 |
| ciod: generated <*> core files for program <*>                                                                 | train        |     342 |
| suppressing further interrupts of same type                                                                    | train        |     252 |

## validation / global: top texts in NEGATIVE windows (15599)
| masked text                                                                                                  | first seen   |   lines |
|:-------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| iar <*> dear <*>                                                                                             | train        | 2022720 |
| CE sym <*>, at <*>, mask <*>                                                                                 | train        |  919138 |
| ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to <*>:<*>: Link has been severed  | train        |  470867 |
| ciod: Error loading <*>: invalid or missing program image, Permission denied                                 | train        |  410704 |
| <*> floating point alignment exceptions                                                                      | train        |  408845 |
| Lustre mount FAILED : <*> : block_id : location                                                              | train        |  247667 |
|                                                                                                              | train        |  239664 |
| ciod: generated <*> core files for program <*>                                                               | train        |  221228 |
| ciod: Error loading <*>: invalid or missing program image, No such file or directory                         | train        |  216179 |
| data cache search parity error detected. attempting to correct                                               | validation   |  198093 |
| ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to <*>:<*>: Link has been severed | train        |  187885 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>                                    | train        |  164821 |
| total of <*> ddr error(s) detected and corrected                                                             | train        |  142256 |
| instruction cache parity error corrected                                                                     | train        |   83952 |
| suppressing further interrupts of same type                                                                  | train        |   64002 |

## validation / global: top UNSEEN (UNK) texts in POSITIVE windows
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| Error receiving packet on tree network, expecting type <*> instead of type <*> (softheader=<*> <*> <*> <*>) PS | validation   |    1024 |
| ddr: activating redundant bit steering for next allocation: rank=<*> symbol=<*>                                | validation   |       4 |
| rts assertion failed: `personality->version == BGLPERSONALITY_VERSION' in `void start()' at <*>:<*>            | validation   |       2 |
| data cache flush parity error detected. attempting to correct                                                  | validation   |       2 |
| ddrSize == <*>*<*>*<*> || ddrSize == <*>*<*>*<*>                                                               | validation   |       1 |

## test / local: top texts in POSITIVE windows (286)
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |     236 |
| ciod: generated <*> core files for program <*>                                                                 | train        |     118 |
| Node card is not fully functional                                                                              | train        |      71 |
| Can not get assembly information for node card                                                                 | train        |      71 |
| Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserte | train        |      71 |
| ciod: Received signal <*>, code=<*>, errno=<*>, address=<*>                                                    | test         |      37 |
| ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to <*>:<*>: Link has been severed    | train        |      32 |
| ciod: Error loading verbose: invalid or missing program image, No such file or directory                       | test         |      24 |
| <*> ddr error(s) detected and corrected on rank <*>, symbol <*> over <*> seconds                               | test         |      23 |
| total of <*> ddr error(s) detected and corrected over <*> seconds                                              | test         |      23 |
| ciod: pollControlDescriptors: Detected the debugger died.                                                      | train        |       8 |
| ciod: Error reading message prefix on CioStream socket to <*>:<*>, Link has been severed                       | test         |       8 |
| <*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected                                            | train        |       6 |
| <*> L3 EDRAM error(s) (dcr <*>) detected and corrected over <*> seconds                                        | test         |       6 |
| critical input interrupt (unit=<*> bit=<*>): warning for torus z- wire, suppressing further interrupts of same | test         |       5 |

## test / local: top texts in NEGATIVE windows (17028)
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| ciod: Error reading message prefix on CioStream socket to <*>:<*>, Link has been severed                       | test         |   11865 |
| iar <*> dear <*>                                                                                               | train        |    8192 |
| ciod: Received signal <*>, code=<*>, errno=<*>, address=<*>                                                    | test         |    6693 |
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |    3935 |
| MACHINE CHECK DCR read timeout (mc=e08x iar <*> lr <*>)                                                        | test         |    3649 |
| <*> total interrupts. <*> critical input interrupts. <*> microseconds total spent on critical input interrupts | test         |    2935 |
| <*> microseconds spent in the rbs signal handler during <*> calls. <*> microseconds was the maximum time for a | test         |    2934 |
| Kernel detected <*> integer alignment exceptions (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*> | test         |    2642 |
| instruction cache parity error corrected                                                                       | train        |    2550 |
| Lustre mount FAILED : <*> : block_id : location                                                                | train        |    1500 |
| ciod: Error loading <*>: invalid or missing program image, Permission denied                                   | train        |    1135 |
| ciod: generated <*> core files for program <*>                                                                 | train        |    1101 |
| <*> ddr error(s) detected and corrected on rank <*>, symbol <*> over <*> seconds                               | test         |    1098 |
| total of <*> ddr error(s) detected and corrected over <*> seconds                                              | test         |    1095 |
| ciod: Error reading message prefix on CioStream socket to <*>:<*>, Connection timed out                        | test         |    1053 |

## test / local: top UNSEEN (UNK) texts in POSITIVE windows
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| ciod: Received signal <*>, code=<*>, errno=<*>, address=<*>                                                    | test         |      37 |
| <*> ddr error(s) detected and corrected on rank <*>, symbol <*> over <*> seconds                               | test         |      23 |
| total of <*> ddr error(s) detected and corrected over <*> seconds                                              | test         |      23 |
| ciod: Error reading message prefix on CioStream socket to <*>:<*>, Link has been severed                       | test         |       8 |
| <*> L3 EDRAM error(s) (dcr <*>) detected and corrected over <*> seconds                                        | test         |       6 |
| critical input interrupt (unit=<*> bit=<*>): warning for torus z- wire, suppressing further interrupts of same | test         |       5 |
| critical input interrupt (unit=<*> bit=<*>): warning for torus z+ wire, suppressing further interrupts of same | test         |       5 |
| PrepareForService is being done on this Midplane (mLctn(R21-M1), mCardSernum(<*>), iWhichCardsToPwrOff(<*>)) b | test         |       5 |
| PrepareForService shutting down Node card(mLctn(<*>), mCardSernum(<*>), mLp(FF:F2:9F:<*>:CE:<*>:<*>:0D:<*>:E9: | test         |       5 |
| PrepareForService shutting down Node card(mLctn(<*>), mCardSernum(<*>), mLp(FF:F2:9F:<*>:C2:D9:<*>:0D:<*>:E9:3 | test         |       5 |
| PrepareForService shutting down Node card(mLctn(<*>), mCardSernum(<*>), mLp(FF:F2:9F:<*>:C2:B6:<*>:0D:<*>:E9:3 | test         |       5 |
| PrepareForService shutting down Node card(mLctn(<*>), mCardSernum(<*>), mLp(FF:F2:9F:<*>:C0:<*>:<*>:0D:<*>:E9: | test         |       5 |
| PrepareForService shutting down Node card(mLctn(<*>), mCardSernum(<*>), mLp(FF:F2:9F:<*>:CF:D7:<*>:0D:<*>:E9:< | test         |       5 |
| PrepareForService shutting down Node card(mLctn(<*>), mCardSernum(<*>), mLp(FF:F2:9F:<*>:C1:<*>:<*>:0D:<*>:E9: | test         |       5 |
| PrepareForService shutting down Node card(mLctn(<*>), mCardSernum(<*>), mLp(FF:F2:9F:<*>:C1:AF:<*>:0D:<*>:E9:3 | test         |       5 |

## test / global: top texts in POSITIVE windows (286)
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| MACHINE CHECK DCR read timeout (mc=e08x iar <*> lr <*>)                                                        | test         |   45784 |
| instruction cache parity error corrected                                                                       | train        |   13043 |
| ciod: generated <*> core files for program <*>                                                                 | train        |    8781 |
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |    5615 |
| ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to <*>:<*>: Link has been severed    | train        |    4784 |
| ciod: Received signal <*>, code=<*>, errno=<*>, address=<*>                                                    | test         |    3737 |
| Ido packet timeout                                                                                             | train        |    1295 |
| total of <*> ddr error(s) detected and corrected over <*> seconds                                              | test         |     778 |
| <*> ddr error(s) detected and corrected on rank <*>, symbol <*> over <*> seconds                               | test         |     775 |
| Node card is not fully functional                                                                              | train        |     713 |
| Can not get assembly information for node card                                                                 | train        |     713 |
| Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserte | train        |     713 |
| <*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected over <*> seconds                           | test         |     627 |
| <*> ddr errors(s) detected and corrected on rank <*>, symbol <*>, bit <*>                                      | train        |     352 |
| critical input interrupt (unit=<*> bit=<*>): warning for tree C1 wire, suppressing further interrupts of same  | test         |     310 |

## test / global: top texts in NEGATIVE windows (17028)
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| iar <*> dear <*>                                                                                               | train        |  837559 |
| ciod: Error reading message prefix on CioStream socket to <*>:<*>, Link has been severed                       | test         |  833334 |
| ciod: Received signal <*>, code=<*>, errno=<*>, address=<*>                                                    | test         |  545688 |
| CE sym <*>, at <*>, mask <*>                                                                                   | train        |  440255 |
| instruction cache parity error corrected                                                                       | train        |  392300 |
| MACHINE CHECK DCR read timeout (mc=e08x iar <*> lr <*>)                                                        | test         |  240132 |
| ciod: generated <*> core files for program <*>                                                                 | train        |  153975 |
| ciod: Error reading message prefix on CioStream socket to <*>:<*>, Connection timed out                        | test         |  117407 |
| total of <*> ddr error(s) detected and corrected over <*> seconds                                              | test         |  117132 |
| <*> ddr error(s) detected and corrected on rank <*>, symbol <*> over <*> seconds                               | test         |  117082 |
|                                                                                                                | train        |  109438 |
| Kernel detected <*> integer alignment exceptions (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*> | test         |  107749 |
| <*> floating point alignment exceptions                                                                        | train        |  101352 |
| ciod: Error loading <*>: invalid or missing program image, Permission denied                                   | train        |   95295 |
| Lustre mount FAILED : <*> : block_id : location                                                                | train        |   93046 |

## test / global: top UNSEEN (UNK) texts in POSITIVE windows
| masked text                                                                                                    | first seen   |   lines |
|:---------------------------------------------------------------------------------------------------------------|:-------------|--------:|
| MACHINE CHECK DCR read timeout (mc=e08x iar <*> lr <*>)                                                        | test         |   45784 |
| ciod: Received signal <*>, code=<*>, errno=<*>, address=<*>                                                    | test         |    3737 |
| total of <*> ddr error(s) detected and corrected over <*> seconds                                              | test         |     778 |
| <*> ddr error(s) detected and corrected on rank <*>, symbol <*> over <*> seconds                               | test         |     775 |
| <*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected over <*> seconds                           | test         |     627 |
| critical input interrupt (unit=<*> bit=<*>): warning for tree C1 wire, suppressing further interrupts of same  | test         |     310 |
| ciod: Error reading message prefix on CioStream socket to <*>:<*>, Link has been severed                       | test         |     263 |
| critical input interrupt (unit=<*> bit=<*>): warning for torus z- wire, suppressing further interrupts of same | test         |     253 |
| <*> L3 EDRAM error(s) (dcr <*>) detected and corrected over <*> seconds                                        | test         |     240 |
| critical input interrupt (unit=<*> bit=<*>): warning for torus y+ wire, suppressing further interrupts of same | test         |     215 |
| critical input interrupt (unit=<*> bit=<*>): warning for torus z+ wire, suppressing further interrupts of same | test         |     203 |
| <*> torus receiver x+ input pipe error(s) (dcr <*>) detected and corrected over <*> seconds                    | test         |     171 |
| <*> torus receiver z+ input pipe error(s) (dcr <*>) detected and corrected over <*> seconds                    | test         |     159 |
| <*> torus receiver z- input pipe error(s) (dcr <*>) detected and corrected over <*> seconds                    | test         |     130 |
| critical input interrupt (unit=<*> bit=<*>): warning for torus x- wire, suppressing further interrupts of same | test         |     116 |
