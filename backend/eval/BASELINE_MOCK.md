
样本数：40   时间容差：±60s
类别分布：activity_poster=11  course_notice=9  homework=9  other=5  repair=6
难度分布：easy=14  hard=12  medium=14
标签覆盖：作业=8  典型场景=18  双时间=4  口语化=1  噪声=5  多时间=1  宣讲会=1  志愿=1  截止=1  报修=1  招新=2  改地点=1  改期=4  放映=1  故障=1  教务=1  无时间=9  易错场景=9  星期=8  晚会=2  比赛=1  灯管=1  热水器=1  相对日期=7  空调=1  绝对日期=3  绝对时间=7  考试=3  补课=1  讲座=2  论文=1  调课=1  跨年=1  边界场景=2  选课=1  门锁=1

分类准确率：37/40 = 92.5%

字段                 P       R      F1    TP    FP    FN
------------------------------------------------------
category        0.93    0.93    0.93    37     3     3
course          0.83    0.77    0.80    10     2     3
location        0.75    0.78    0.77    18     6     5
issuer          0.58    1.00    0.74     7     5     0
event_time      0.67    0.78    0.72    14     7     4
deadline        0.73    0.65    0.69    11     4     6
------------------------------------------------------
微平均             0.78    0.82    0.80    97    27    21

未命中明细（37 项）：
  [cn_02] location: 期望='A203'  实际=None
  [act_02] event_time: 期望='2027-01-10T19:30:00'  实际=datetime.datetime(2027, 1, 5, 9, 0)
  [act_02] deadline: 期望='2027-01-05T23:59:00'  实际=None
  [oth_01] category: 期望='other'  实际='repair'
  [oth_01] location: 期望=None  实际='请同学们注意教室'
  [hw_04] event_time: 期望=None  实际=datetime.datetime(2026, 9, 30, 9, 0)
  [hw_04] deadline: 期望='2026-09-30T23:59:00'  实际=datetime.datetime(2026, 9, 30, 9, 0)
  [hw_05] course: 期望='操作系统'  实际=None
  [hw_06] course: 期望='机器学习'  实际=None
  [hw_06] event_time: 期望=None  实际=datetime.datetime(2026, 9, 8, 12, 0)
  [hw_06] deadline: 期望='2026-09-08T12:00:00'  实际=datetime.datetime(2026, 9, 1, 23, 59)
  [hw_07] course: 期望='概率论'  实际=None
  [hw_09] event_time: 期望=None  实际=datetime.datetime(2026, 9, 20, 9, 0)
  [hw_09] deadline: 期望='2026-09-20T23:59:00'  实际=datetime.datetime(2026, 9, 20, 9, 0)
  [cn_03] location: 期望='A101'  实际=None
  [cn_06] location: 期望='图书馆西侧报告厅'  实际='由A201调整为图书馆'
  [cn_06] issuer: 期望=None  实际='调整为图书馆'
  [cn_07] event_time: 期望=None  实际=datetime.datetime(2026, 8, 29, 9, 0)
  [cn_07] deadline: 期望='2026-08-29T23:59:00'  实际=None
  [cn_09] category: 期望='course_notice'  实际='other'
  [act_04] location: 期望='东体育场'  实际=None
  [act_04] event_time: 期望='2026-09-12T15:00:00'  实际=None
  [act_07] location: 期望='图书馆南门'  实际='00在图书馆'
  [act_07] issuer: 期望=None  实际='在图书馆'
  [act_08] course: 期望=None  实际='大模型时代的软件工程'
  [act_10] event_time: 期望='2026-10-15T07:30:00'  实际=datetime.datetime(2026, 10, 15, 19, 30)
  [act_11] course: 期望=None  实际='流浪地球3'
  [act_11] event_time: 期望='2026-08-24T19:00:00'  实际=None
  [rp_03] issuer: 期望=None  实际='图书馆'
  [rp_03] event_time: 期望=None  实际=datetime.datetime(2026, 8, 25, 14, 0)
  [rp_05] deadline: 期望='2026-08-28T23:59:00'  实际=datetime.datetime(2026, 8, 28, 9, 0)
  [oth_03] location: 期望=None  实际='有同学在第一教学楼'
  [oth_03] issuer: 期望=None  实际='请失主到辅导员'
  [oth_04] location: 期望=None  实际='图书馆'
  [oth_04] issuer: 期望=None  实际='图书馆'
  [oth_05] category: 期望='other'  实际='repair'
  [oth_05] location: 期望=None  实际='实验室'
