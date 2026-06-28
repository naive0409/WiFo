This repository contains the datasets of Channel State Information (CSI) from 11 users for human gait recognition using WiFi. 

## Cite the Paper

Yi Zhang, Yue Zheng, Guidong Zhang, Kun Qian, Chen Qian, Zheng Yang. 2020. "GaitID: Robust Wi-Fi Based Gait Recognition". In Proceedings of International Conference on Wireless Algorithms, Systems, and Applications (WASA'2020). Lecture Notes in Computer Science, vol 12384. Springer, Cham. 

Readers can also check the homepage of project Widar3.0 for more details about WiFi sensing. 

## Dataset Description

<table><tr><td>Data Type</td><td>File Name Format</td><td>Description</td></tr><tr><td>CSI</td><td>id-a-b-Rx.dat</td><td>&#x27;id&#x27;: user&#x27;s id; &#x27;a&#x27;: track no., &#x27;b&#x27;: repetition no., &#x27;Rx&#x27;: Wi-Fi receiver id.</td></tr></table>

Note: For repetition no., the odd sequence number refer to the forward directions and the even sequence number refer to the reverse directions. Track no., Rx location and forward direction refer to Device Deployment part. 

<table><tr><td>Room #</td><td>ID</td><td>Date</td><td>Instances</td></tr><tr><td>Room #1</td><td>user1</td><td>20190627</td><td>4*100</td></tr><tr><td>Room #1</td><td>user2</td><td>20190627</td><td>4*100</td></tr><tr><td>Room #1</td><td>user3</td><td>20190706</td><td>4*100</td></tr><tr><td>Room #1</td><td>user4</td><td>20190707</td><td>4*100</td></tr><tr><td>Room #1</td><td>user5</td><td>20190713</td><td>4*100</td></tr><tr><td>Room #1</td><td>user6</td><td>20190713</td><td>4*100</td></tr><tr><td>Room #1</td><td>user7</td><td>20190713</td><td>4*100</td></tr><tr><td>Room #1</td><td>user8</td><td>20190718</td><td>4*50</td></tr><tr><td>Room #1</td><td>user9</td><td>20190718</td><td>4*50</td></tr><tr><td>Room #1</td><td>user10</td><td>20190719</td><td>4*50</td></tr><tr><td>Room #2</td><td>user11</td><td>20190719</td><td>4*50</td></tr><tr><td>Room #2</td><td>user1</td><td>20190719</td><td>4*20</td></tr><tr><td>Room #2</td><td>user2</td><td>20190719</td><td>4*20</td></tr></table>

Room#1 - Classroom 

![](images/1d3e42c5dad9fb7f1a32dfba4777d38e73de2fb6ff0353daef137bb6d812b42c.jpg)



(a) Discussion Room



Room#2 - Hall


![](images/546e0d031d40eba6fa539cb93ea33d7531cc3bd5608f9cd8995d91b1cd44e945.jpg)



Device Deployment


![](images/3e9e3122de1a58964ebbca20b467b57f8112d75306cb0b00da6a9d6ed3ebcaeb.jpg)


<table><tr><td>Track #</td><td>Location/m (forward direction)</td></tr><tr><td>1</td><td>(2.3, 0.55) =&gt; (2.3, 4.95)</td></tr><tr><td>2</td><td>(0.575, 0.55) =&gt; (4.025, 3.85)</td></tr><tr><td>3</td><td>(0.575, 2.2) =&gt; (5.175, 2.2)</td></tr><tr><td>4</td><td>(1.15, 4.4) =&gt; (4.6, 1.1)</td></tr></table>


Note: Tx location: (0, 0), Rx-1 location: (1.7, -0.5), Rx-2 location: (3.4, -0.5), Rx-3 location: (4.6, 0), Rx-4 location: (-0.55, 1.65), Rx-5 location: (-0.55, 3.3), Rx-6 location: (0, 4.4).
