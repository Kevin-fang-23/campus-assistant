
样本数：40   时间容差：±60s
类别分布：activity_poster=11  course_notice=9  homework=9  other=5  repair=6
难度分布：easy=14  hard=12  medium=14
标签覆盖：作业=8  典型场景=18  双时间=4  口语化=1  噪声=5  多时间=1  宣讲会=1  志愿=1  截止=1  报修=1  招新=2  改地点=1  改期=4  放映=1  故障=1  教务=1  无时间=9  易错场景=9  星期=8  晚会=2  比赛=1  灯管=1  热水器=1  相对日期=7  空调=1  绝对日期=3  绝对时间=7  考试=3  补课=1  讲座=2  论文=1  调课=1  跨年=1  边界场景=2  选课=1  门锁=1

分类准确率：40/40 = 100.0%

字段                 P       R      F1    TP    FP    FN
------------------------------------------------------
category        1.00    1.00    1.00    40     0     0
course          0.87    1.00    0.93    13     2     0
location        0.82    1.00    0.90    23     5     0
issuer          0.47    1.00    0.64     7     8     0
event_time      0.62    0.83    0.71    15     9     3
deadline        1.00    1.00    1.00    17     0     0
------------------------------------------------------
微平均             0.83    0.97    0.89   115    24     3

未命中明细（24 项）：
  [act_03] event_time: 期望='2026-09-02T19:30:00'  实际=datetime.datetime(2026, 8, 27, 19, 30)
  [rp_01] event_time: 期望=None  实际=datetime.datetime(2026, 8, 25, 10, 0)
  [oth_01] location: 期望=None  实际='请同学们注意教室'
  [hw_04] location: 期望=None  实际='学习通'
  [hw_04] event_time: 期望=None  实际=datetime.datetime(2026, 9, 30, 9, 0)
  [hw_06] event_time: 期望=None  实际=datetime.datetime(2026, 9, 8, 12, 0)
  [hw_09] event_time: 期望=None  实际=datetime.datetime(2026, 9, 20, 9, 0)
  [cn_06] issuer: 期望=None  实际='调整为图书馆'
  [cn_07] event_time: 期望=None  实际=datetime.datetime(2026, 8, 29, 9, 0)
  [cn_09] event_time: 期望='2026-08-25T19:30:00'  实际=datetime.datetime(2026, 8, 26, 19, 30)
  [act_06] issuer: 期望=None  实际='腾讯公司'
  [act_07] issuer: 期望=None  实际='在图书馆'
  [act_07] event_time: 期望='2026-08-29T08:00:00'  实际=datetime.datetime(2026, 8, 30, 8, 0)
  [act_08] course: 期望=None  实际='大模型时代的软件工程'
  [act_09] issuer: 期望=None  实际='摄影协会'
  [act_10] issuer: 期望=None  实际='未知'
  [act_11] course: 期望=None  实际='流浪地球3'
  [rp_03] issuer: 期望=None  实际='图书馆'
  [rp_03] event_time: 期望=None  实际=datetime.datetime(2026, 8, 25, 14, 0)
  [oth_03] location: 期望=None  实际='第一教学楼辅导员办公室'
  [oth_03] issuer: 期望=None  实际='请失主到辅导员'
  [oth_04] location: 期望=None  实际='图书馆'
  [oth_04] issuer: 期望=None  实际='图书馆'
  [oth_05] location: 期望=None  实际='实验室'
