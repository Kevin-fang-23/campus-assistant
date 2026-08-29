"""种子数据：python -m app.seed
覆盖四类场景 + 一条近似重复通知（用于验证 FAISS 去重）。
"""
from __future__ import annotations

from .db import SessionLocal, init_db
from .models import Category
from .services.ingest import ingest_text
from .services.vector_store import get_store

SAMPLES: list[tuple[str, str]] = [
    (
        "course_notice.txt",
        """关于《计算机网络》课程调课的通知
各位同学：
因主讲教师外出参加学术会议，原定本周三下午的《计算机网络》课程调整至本周五下午2:30，
地点改为第三教学楼A305。请各位同学准时到场，勿缺勤。
教务处
联系电话：027-87654321""",
    ),
    (
        "homework.txt",
        """《数据结构》第三次实验报告作业要求
1. 实现二叉搜索树的插入、删除、查找，并给出复杂度分析；
2. 提交内容包括源代码与实验报告 PDF，命名格式 学号_姓名_实验三；
3. 提交方式：学习通课程平台。
截止时间：9月10日 23:00，逾期不计分。查重率需低于 15%。
助教邮箱：ta_ds@campus.edu.cn""",
    ),
    (
        "activity_poster.txt",
        """人工智能前沿讲座 · 大模型与多模态
主办：计算机学院 学生会
主讲嘉宾：李某某 研究员
时间：明天晚上7点
地点：图书馆报告厅
报名截止：今天下午5点，扫码进QQ群 872341905 领取入场资格，免费入场。""",
    ),
    (
        "repair.txt",
        """宿舍报修申请
报修位置：3号楼412室
故障描述：卫生间水管漏水，地面积水严重，热水器无法出热水。
希望后勤维修师傅明天上午10点前上门处理。
联系人手机：13800001234""",
    ),
    (
        "course_notice_dup.txt",
        """关于《计算机网络》课程调课的通知
各位同学：因主讲教师外出参加学术会议，原定本周三下午的《计算机网络》课程调整到本周五下午2:30，
地点改为第三教学楼A305，请准时到场。
教务处""",
    ),
]


def run() -> None:
    init_db()
    with SessionLocal() as db:
        get_store().load_from_db(db)
        for filename, content in SAMPLES:
            result = ingest_text(db, content, filename)
            notice = result.notice
            label = Category.LABELS.get(notice.category, notice.category) if notice else "-"
            print(f"\n=== {filename} → {label} (conf={notice.confidence if notice else 0:.2f}) ===")
            if notice:
                print(f"  标题: {notice.title}")
                print(f"  时间: event={notice.event_time} deadline={notice.deadline}")
                print(f"  地点: {notice.location} | 发布方: {notice.issuer} | 课程: {notice.course}")
                print(f"  联系: {notice.contacts}")
            print(f"  待办: {[t.title for t in result.tasks]}")
            if result.duplicate:
                print(f"  ⚠ 判定与通知 #{result.duplicate_of_id} 重复")
            for w in result.warnings:
                print(f"  warn: {w}")


if __name__ == "__main__":
    run()
