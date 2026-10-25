# Feature importance — XGBoost content model (tag='primary')

Gain-based importance; direction = mean feature value in positive vs negative VALIDATION windows.

## N=1 min — content model (54 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| [global] mmcs db server has been started <*> --useDatabase BGL --dbproperties <*> --iolog <*> --rec |   50   |               0     |               0     |
| [global] PrepareForService shutting down <*> mCardSernum(<*>) mLp(FF F2 9F <*> <*> <*> <*> 0D <*> < |   40.8 |               0.003 |               0.022 |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   27.9 |               0     |               0     |
| dow_5                                                                                               |   25.7 |               0.171 |               0.138 |
| [global] While initializing service card iDo with LP of FF F2 9F <*> <*> <*> <*> 0D <*> EA <*> <*>  |   23   |               0     |               0     |
| [global] <*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected                        |   22.6 |               0.839 |               0.654 |
| n_local                                                                                             |   20.2 |               1.296 |               4.934 |
| [global] total of <*> ddr error(s) detected and corrected                                           |   19.2 |               1.119 |               1.075 |
| [global] ciodb has been restarted.                                                                  |   18.2 |               0.003 |               0.03  |
| mean_dt_local                                                                                       |   18.1 |            3155.79  |            2849.57  |
| [global] Node card is not fully functional                                                          |   17.7 |               0.056 |               0.092 |
| [global] Target ido <*> F2 9F <*> E6 <*> <*> 0D <*> E9 <*> FB/JTAG/<*> Message JtagId All all zeros |   17.7 |               0     |               0     |
| [local] Node card is not fully functional                                                           |   17.4 |               0     |               0.001 |
| hour_sin                                                                                            |   16.8 |               0.29  |               0.021 |
| [global] MidplaneSwitchController sendTrain() port disconnected R07-M1-L1-U19-E                     |   15.1 |               0     |               0     |
| [local] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found <  |   14.9 |               0     |               0.001 |
| [global] suppressing further interrupts of same type                                                |   14.8 |               0.383 |               0.33  |
| [local] ciod Error loading <*> invalid or missing program image No such file or directory           |   14.8 |               0     |               0.023 |
| [global] critical input interrupt (unit <*> bit <*>) warning for <*> <*> wire                       |   14.4 |               0.35  |               0.316 |
| [local] ciod Message code <*> is not <*> or <*>                                                     |   14   |               0     |               0     |
| n_global                                                                                            |   14   |             250.618 |             595.805 |
| [global] <*> torus receiver <*> input pipe error(s) (dcr <*>) detected and corrected                |   13.6 |               0.717 |               0.629 |
| n_nontarget_alerts_global                                                                           |   13.5 |               0     |              57.878 |
| [global] LinkCard power module U58 is not accessible                                                |   13.4 |               0     |               0     |
| weekend                                                                                             |   13   |               0.377 |               0.293 |

Share of total gain by feature group: time 10.2%, volume 7.3%, summary 9.6%, history 0.0%, counts_local 11.0%, counts_global 61.9%

## N=1 min — full model (55 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| [global] mmcs db server has been started <*> --useDatabase BGL --dbproperties <*> --iolog <*> --rec |   53.8 |               0     |               0     |
| dow_5                                                                                               |   36.3 |               0.171 |               0.138 |
| [global] Node card is not fully functional                                                          |   35.9 |               0.056 |               0.092 |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   30.4 |               0     |               0     |
| [global] While initializing service card iDo with LP of FF F2 9F <*> <*> <*> <*> 0D <*> EA <*> <*>  |   27   |               0     |               0     |
| unit_fails_7d                                                                                       |   24.2 |               0.251 |               0.163 |
| hrs_since_unit_fail                                                                                 |   23.8 |             147.238 |             157.081 |
| [global] <*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected                        |   23.4 |               0.839 |               0.654 |
| [global] ciodb has been restarted.                                                                  |   21.3 |               0.003 |               0.03  |
| [global] total of <*> ddr error(s) detected and corrected                                           |   20.5 |               1.119 |               1.075 |
| mass_24h                                                                                            |   20.1 |               0.05  |               0.055 |
| mean_dt_local                                                                                       |   18.7 |            3155.79  |            2849.57  |
| hrs_since_sys_fail                                                                                  |   18.4 |               7.982 |              11.33  |
| [global] MidplaneSwitchController sendTrain() port disconnected R07-M1-L1-U19-E                     |   17   |               0     |               0     |
| [local] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found <  |   16.9 |               0     |               0.001 |
| n_nontarget_alerts_global                                                                           |   16.9 |               0     |              57.878 |
| [local] ciod Error loading <*> invalid or missing program image No such file or directory           |   16.3 |               0     |               0.023 |
| max_repeat_global                                                                                   |   15.9 |             112.829 |             191.819 |
| [global] suppressing further interrupts of same type                                                |   14.8 |               0.383 |               0.33  |
| [global] Target ido <*> F2 9F <*> E6 <*> <*> 0D <*> E9 <*> FB/JTAG/<*> Message JtagId All all zeros |   14.5 |               0     |               0     |
| [global] ddr activating redundant bit steering rank <*> symbol <*>                                  |   14.2 |               0.359 |               0.202 |
| unit_nontarget_24h                                                                                  |   13.5 |               0.925 |               1.858 |
| [global] <*> ddr errors(s) detected and corrected on rank <*> symbol <*> bit <*>                    |   13.4 |               1.185 |               1.103 |
| has_nontarget_alert_local                                                                           |   13.1 |               0     |               0.096 |
| n_global                                                                                            |   13.1 |             250.618 |             595.805 |

Share of total gain by feature group: time 9.9%, volume 5.0%, summary 10.5%, history 15.8%, counts_local 6.0%, counts_global 52.8%

## N=5 min — content model (84 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| dow_2                                                                                               |   35.8 |               0.052 |               0.12  |
| dow_1                                                                                               |   34.5 |               0.155 |               0.128 |
| [global] Target ido <*> F2 9F <*> E8 1A <*> 0D <*> E9 <*> E5/JTAG/<*> Message ScomError             |   33   |               0     |               0     |
| [global] Target ido <*> F2 9F <*> E6 <*> <*> 0D <*> E9 <*> <*>/JTAG/<*> Message Pll failed to lock  |   32.1 |               0     |               0     |
| [global] While initializing service card iDo with LP of FF F2 9F <*> <*> <*> <*> 0D <*> EA <*> <*>  |   31.8 |               0     |               0     |
| [global] idoproxydb hit ASSERT condition ASSERT expression <*> Source file <*> Source line <*> Func |   31.2 |               0     |               0     |
| [global] Ido packet timeout                                                                         |   29.2 |               0.011 |               0.003 |
| log_n_global                                                                                        |   28.8 |               2.862 |               3.45  |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   27.1 |               0     |               0     |
| [global] EndServiceAction <*> performed upon <*> by <*>                                             |   25.5 |               0.025 |               0.021 |
| [local] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found <  |   25.2 |               0     |               0.001 |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   25   |               0.096 |               0.09  |
| [local] ciod cpu <*> at treeaddr <*> sent unrecognized message <*>                                  |   23.6 |               0     |               0     |
| [local] EndServiceAction <*> performed upon <*> by <*>                                              |   23.3 |               0     |               0     |
| n_global                                                                                            |   22.2 |             218.104 |             597.907 |
| [local] Node card is not fully functional                                                           |   22   |               0     |               0.001 |
| [global] EndServiceAction is restarting the <*> in midplane <*> as part of Service Action <*>       |   22   |               0.025 |               0.023 |
| [global] MailboxMonitor serviceMailboxes() lib ido error -<*> unexpected socket error Broken pipe   |   21.9 |               0     |               0     |
| [global] instruction cache parity error corrected                                                   |   21.7 |               0.927 |               0.588 |
| [global] ddr activating redundant bit steering rank <*> symbol <*>                                  |   21.2 |               0.306 |               0.202 |
| dow_4                                                                                               |   20.7 |               0.062 |               0.146 |
| max_repeat_global                                                                                   |   20.4 |              92.549 |             192.33  |
| [local] ciod Message code <*> is not <*> or <*>                                                     |   19.9 |               0     |               0     |
| [global] BGLMaster has been started <*> --consoleip <*> --consoleport <*> --configfile bglmaster.in |   19.5 |               0.004 |               0.03  |
| [local] ciod Error loading <*> invalid or missing program image No such file or directory           |   19.3 |               0.027 |               0.023 |

Share of total gain by feature group: time 13.5%, volume 9.5%, summary 6.4%, history 0.0%, counts_local 18.5%, counts_global 52.2%

## N=5 min — full model (85 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| mass_24h                                                                                            |   59.7 |               0.093 |               0.054 |
| sys_fails_24h                                                                                       |   40   |               3.036 |               3.054 |
| [global] Ido packet timeout                                                                         |   33.1 |               0.011 |               0.003 |
| [global] Target ido <*> F2 9F <*> E6 <*> <*> 0D <*> E9 <*> <*>/JTAG/<*> Message Pll failed to lock  |   32.7 |               0     |               0     |
| [global] Target ido <*> F2 9F <*> E8 1A <*> 0D <*> E9 <*> E5/JTAG/<*> Message ScomError             |   32.3 |               0     |               0     |
| [global] idoproxydb hit ASSERT condition ASSERT expression <*> Source file <*> Source line <*> Func |   31.3 |               0     |               0     |
| unit_fails_7d                                                                                       |   31.1 |               0.228 |               0.163 |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   30   |               0     |               0     |
| [global] While initializing service card iDo with LP of FF F2 9F <*> <*> <*> <*> 0D <*> EA <*> <*>  |   28.6 |               0     |               0     |
| n_global                                                                                            |   28.6 |             218.104 |             597.907 |
| hrs_since_unit_fail                                                                                 |   28.4 |             149.391 |             157.105 |
| [global] NodeCard temperature sensor chip <*> is not accessible                                     |   27.4 |               0     |               0     |
| unit_fails_24h                                                                                      |   25.7 |               0.067 |               0.022 |
| [local] Can not get assembly information for node card                                              |   25.5 |               0     |               0.001 |
| [global] Target ido <*> F2 9F <*> E6 <*> <*> 0D <*> E9 <*> FB/JTAG/<*> Message JtagId All all zeros |   25.3 |               0     |               0     |
| [global] Node card is not fully functional                                                          |   24.7 |               0.096 |               0.091 |
| [global] Can not get assembly information for node card                                             |   24.2 |               0.096 |               0.091 |
| [global] EndServiceAction is restarting the <*> in midplane <*> as part of Service Action <*>       |   23.7 |               0.025 |               0.023 |
| log_n_global                                                                                        |   23.4 |               2.862 |               3.45  |
| [global] MailboxMonitor serviceMailboxes() lib ido error -<*> unexpected socket error Broken pipe   |   23   |               0     |               0     |
| hrs_since_sys_fail                                                                                  |   21.9 |               7.589 |              11.351 |
| [local] critical input interrupt (unit <*> bit <*>) warning for <*> <*> wire                        |   21.2 |               0     |               0.021 |
| [global] EndServiceAction <*> performed upon <*> by <*>                                             |   20.6 |               0.025 |               0.021 |
| [global] ciodb has been restarted.                                                                  |   20.1 |               0.004 |               0.03  |
| [local] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found <  |   19.9 |               0     |               0.001 |

Share of total gain by feature group: time 6.5%, volume 8.1%, summary 5.8%, history 15.9%, counts_local 11.2%, counts_global 52.5%

## N=15 min — content model (110 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| [global] Target ido <*> F2 9F <*> E6 <*> <*> 0D <*> E9 <*> <*>/JTAG/<*> Message Pll failed to lock  |   53.8 |               0     |               0     |
| [global] Target ido <*> F2 9F <*> E8 1A <*> 0D <*> E9 <*> E5/JTAG/<*> Message ScomError             |   51.7 |               0     |               0     |
| [global] NodeCard VPD chip is not accessible                                                        |   41.7 |               0     |               0     |
| [global] NodeCard temperature sensor chip <*> is not accessible                                     |   39.9 |               0     |               0     |
| dow_2                                                                                               |   31.9 |               0.077 |               0.12  |
| [global] ciodb has been restarted.                                                                  |   28.2 |               0     |               0.03  |
| [global] EndServiceAction <*> performed upon <*> by <*>                                             |   19.8 |               0.025 |               0.021 |
| [local] MidplaneSwitchController receiveTrain() iap failed <*> status <*> <*>                       |   19.6 |               0     |               0     |
| [global] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found < |   18.7 |               0.052 |               0.073 |
| [global] EndServiceAction is restarting the <*> in midplane <*> as part of Service Action <*>       |   18.5 |               0.025 |               0.023 |
| [local] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT  |   17.9 |               0     |               0.001 |
| [local] Node card is not fully functional                                                           |   17.9 |               0     |               0.001 |
| dow_5                                                                                               |   17.6 |               0.128 |               0.138 |
| [global] idoproxydb has been started <*> Input parameters -enableflush -loguserinfo <*> BlueGene1   |   17   |               0     |               0.03  |
| [local] Can not get assembly information for node card                                              |   16.7 |               0     |               0.001 |
| [global] BGLMaster has been started <*> --consoleip <*> --consoleport <*> --configfile bglmaster.in |   16.6 |               0     |               0.03  |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   16.3 |               0.088 |               0.09  |
| [global] Lustre mount FAILED <*> block id location                                                  |   16.2 |               0     |               0.186 |
| dow_6                                                                                               |   16.2 |               0.224 |               0.154 |
| [global] <*> <*> <*> error(s) (dcr <*>) detected and corrected                                      |   15.9 |               0.496 |               0.409 |
| hour_cos                                                                                            |   15.7 |              -0.163 |              -0.042 |
| [local] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found <  |   15   |               0     |               0.001 |
| [global] Starting SystemController                                                                  |   14.9 |               0     |               0.004 |
| [global] idoproxydb hit ASSERT condition ASSERT expression <*> Source file <*> Source line <*> Func |   14.9 |               0     |               0     |
| [global] <*> ddr errors(s) detected and corrected on rank <*> symbol <*> bit <*>                    |   14.9 |               1.073 |               1.104 |

Share of total gain by feature group: time 11.2%, volume 4.8%, summary 4.8%, history 0.0%, counts_local 15.2%, counts_global 64.0%

## N=15 min — full model (126 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| [global] NodeCard VPD chip is not accessible                                                        |   71.2 |               0     |               0     |
| [global] Target ido <*> F2 9F <*> E8 1A <*> 0D <*> E9 <*> E5/JTAG/<*> Message ScomError             |   51.3 |               0     |               0     |
| [global] NodeCard temperature sensor chip <*> is not accessible                                     |   50.1 |               0     |               0     |
| [global] Target ido <*> F2 9F <*> E6 <*> <*> 0D <*> E9 <*> <*>/JTAG/<*> Message Pll failed to lock  |   48.1 |               0     |               0     |
| [global] While initializing node card ido with LP of FF F2 9F <*> E6 <*> <*> 0D <*> E9 <*> <*> caug |   44.5 |               0     |               0     |
| unit_fails_24h                                                                                      |   30.4 |               0.051 |               0.022 |
| [global] ciodb has been restarted.                                                                  |   26.2 |               0     |               0.03  |
| dow_5                                                                                               |   25.6 |               0.128 |               0.138 |
| unit_fails_7d                                                                                       |   23.9 |               0.184 |               0.163 |
| [local] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT  |   22.6 |               0     |               0.001 |
| [global] While initializing node card ido with LP of FF F2 9F <*> <*> <*> <*> 0D <*> E9 <*> <*> cau |   20.7 |               0     |               0     |
| mass_24h                                                                                            |   20.6 |               0.061 |               0.054 |
| [global] instruction address <*>                                                                    |   20.2 |               0.032 |               0.116 |
| hrs_since_unit_fail                                                                                 |   19.9 |             153.787 |             157.148 |
| [global] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found < |   19   |               0.052 |               0.073 |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   19   |               0.088 |               0.09  |
| [local] Can not get assembly information for node card                                              |   18.9 |               0     |               0.001 |
| [global] BGLMaster has been started <*> --consoleip <*> --consoleport <*> --configfile bglmaster.in |   18.2 |               0     |               0.03  |
| [global] EndServiceAction is restarting the <*> in midplane <*> as part of Service Action <*>       |   17.8 |               0.025 |               0.023 |
| [local] Node card is not fully functional                                                           |   17.5 |               0     |               0.001 |
| [global] idoproxydb has been started <*> Input parameters -enableflush -loguserinfo <*> BlueGene1   |   17.2 |               0     |               0.03  |
| [global] mmcs db server has been started <*> --useDatabase BGL --dbproperties <*> --iolog <*> --rec |   16.9 |               0     |               0     |
| dow_2                                                                                               |   16.7 |               0.077 |               0.12  |
| [global] Lustre mount FAILED <*> block id location                                                  |   16.1 |               0     |               0.186 |
| unit_fails_6h                                                                                       |   16   |               0.046 |               0.005 |

Share of total gain by feature group: time 8.0%, volume 4.3%, summary 4.0%, history 10.6%, counts_local 10.7%, counts_global 62.4%

## N=30 min — content model (61 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| [global] EndServiceAction <*> performed upon <*> by <*>                                             |   64   |               0.011 |               0.022 |
| [global] NodeCard temperature sensor chip <*> is not accessible                                     |   44.2 |               0     |               0     |
| [global] BGLMaster has been started <*> --consoleip <*> --consoleport <*> --configfile bglmaster.in |   27.2 |               0     |               0.031 |
| [local] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found <  |   26.2 |               0     |               0.001 |
| dow_2                                                                                               |   25.5 |               0.119 |               0.12  |
| [global] ciod Error creating node map from file <*> <*> <*>                                         |   25.3 |               0     |               0     |
| n_target_lines_global                                                                               |   25.3 |              15.973 |               2.749 |
| max_repeat_local                                                                                    |   25.1 |               0.588 |               3.125 |
| [global] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found < |   22.9 |               0.017 |               0.074 |
| [global] Lustre mount FAILED <*> block id location                                                  |   22.7 |               0.008 |               0.189 |
| [global] MailboxMonitor serviceMailboxes() lib ido error -<*> socket closed                         |   21.4 |               0     |               0.003 |
| [global] NodeCard power module <*> is not accessible                                                |   20.7 |               0     |               0.007 |
| [global] While initializing service card iDo with LP of FF F2 9F <*> <*> <*> <*> 0D <*> EA <*> <*>  |   20.7 |               0     |               0     |
| [global] Node card is not fully functional                                                          |   20.6 |               0.043 |               0.092 |
| n_units_active_global                                                                               |   20.1 |              15.462 |              26.501 |
| [global] rts bad message header cpu <*> invalid (softheader <*> <*> <*> <*>) PSR0 <*> PSR1 <*> PRXF |   20   |               0     |               0     |
| [global] Error unable to mount filesystem                                                           |   19.4 |               0     |               0     |
| [global] machine check <*>                                                                          |   19.3 |               0.058 |               0.107 |
| distinct_global                                                                                     |   19.2 |               5.735 |               6.924 |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   19   |               0     |               0     |
| hour_cos                                                                                            |   18.8 |              -0.141 |              -0.041 |
| [global] critical input interrupt (unit <*> bit <*>) warning for <*> <*> wire                       |   18.6 |               0.303 |               0.317 |
| [local] ciod Error opening node map file <*> No such file or directory                              |   18.6 |               0     |               0     |
| max_repeat_global                                                                                   |   18.6 |             170.265 |             192.832 |
| hour_sin                                                                                            |   17.9 |               0.241 |               0.013 |

Share of total gain by feature group: time 13.3%, volume 8.7%, summary 11.6%, history 0.0%, counts_local 4.9%, counts_global 61.6%

## N=30 min — full model (69 features used)

| feature                                                                                             |   gain |   mean in positives |   mean in negatives |
|:----------------------------------------------------------------------------------------------------|-------:|--------------------:|--------------------:|
| [global] EndServiceAction <*> performed upon <*> by <*>                                             |   71.4 |               0.011 |               0.022 |
| [global] NodeCard temperature sensor chip <*> is not accessible                                     |   48.2 |               0     |               0     |
| dow_6                                                                                               |   38.6 |               0.196 |               0.154 |
| hrs_since_unit_fail                                                                                 |   37.8 |             147.156 |             157.319 |
| max_repeat_local                                                                                    |   37.7 |               0.588 |               3.125 |
| [local] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found <  |   34.9 |               0     |               0.001 |
| unit_fails_7d                                                                                       |   33.9 |               0.231 |               0.161 |
| [global] Lustre mount FAILED <*> block id location                                                  |   31.7 |               0.008 |               0.189 |
| [global] ciod Error creating node map from file <*> <*> <*>                                         |   31   |               0     |               0     |
| hour_cos                                                                                            |   30.6 |              -0.141 |              -0.041 |
| mass_24h                                                                                            |   30   |               0.031 |               0.055 |
| [global] BGLMaster has been started <*> --consoleip <*> --consoleport <*> --configfile bglmaster.in |   29.3 |               0     |               0.031 |
| [global] Node card status no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JT |   27.9 |               0     |               0     |
| distinct_global                                                                                     |   26.3 |               5.735 |               6.924 |
| [global] Node card VPD check <*> node in processor card slot <*> do not match. VPD ecid <*> found < |   25.9 |               0.017 |               0.074 |
| [global] Error unable to mount filesystem                                                           |   25.4 |               0     |               0     |
| [global] While initializing service card iDo with LP of FF F2 9F <*> <*> <*> <*> 0D <*> EA <*> <*>  |   23.9 |               0     |               0     |
| [global] MailboxMonitor serviceMailboxes() lib ido error -<*> socket closed                         |   23.8 |               0     |               0.003 |
| dow_0                                                                                               |   23.4 |               0.215 |               0.15  |
| hrs_since_sys_fail                                                                                  |   23.2 |              10.966 |              11.379 |
| [global] Node card is not fully functional                                                          |   22.9 |               0.043 |               0.092 |
| n_target_lines_global                                                                               |   22.8 |              15.973 |               2.749 |
| [global] NodeCard power module <*> is not accessible                                                |   22.6 |               0     |               0.007 |
| n_global                                                                                            |   21.3 |             413.662 |             598.911 |
| dow_2                                                                                               |   21.3 |               0.119 |               0.12  |

Share of total gain by feature group: time 14.9%, volume 6.4%, summary 10.6%, history 11.7%, counts_local 7.0%, counts_global 49.3%
