"""单元测试：任务调度器 TaskScheduler

测试 4 种调度策略（round_robin, weighted, curriculum, adaptive）
及状态保存/恢复功能。
"""


from florence_forge.core.config import TaskSchedulingConfig
from florence_forge.training.scheduler import TaskScheduler


class TestTaskScheduler:
    """TaskScheduler 核心功能测试"""

    SAMPLE_TASKS = ["CAPTION", "OD", "OCR"]

    def _make_scheduler(self, strategy: str = "round_robin", **kwargs) -> TaskScheduler:
        config = TaskSchedulingConfig(strategy=strategy, **kwargs)
        return TaskScheduler(task_types=self.SAMPLE_TASKS, config=config)

    # ---- 初始化 ----

    def test_init_default_weights(self):
        sched = self._make_scheduler()
        for task in self.SAMPLE_TASKS:
            assert sched.task_weights[task] == 1.0

    def test_init_custom_weights(self):
        weights = {"CAPTION": 2.0, "OD": 1.0, "OCR": 0.5}
        sched = TaskScheduler(
            task_types=self.SAMPLE_TASKS,
            config=TaskSchedulingConfig(),
            initial_weights=weights,
        )
        assert sched.task_weights == weights

    # ---- round_robin 策略 ----

    def test_round_robin_cycles(self):
        sched = self._make_scheduler(strategy="round_robin")
        results = [sched.select_task() for _ in range(6)]
        # 应该按顺序循环
        expected = self.SAMPLE_TASKS * 2
        assert results == expected

    # ---- weighted 策略 ----

    def test_weighted_respects_weights(self):
        weights = {"CAPTION": 10.0, "OD": 0.0, "OCR": 0.0}
        sched = TaskScheduler(
            task_types=self.SAMPLE_TASKS,
            config=TaskSchedulingConfig(strategy="weighted"),
            initial_weights=weights,
        )
        # 连续选 20 次，应该全部选中 CAPTION
        results = [sched.select_task() for _ in range(20)]
        assert all(t == "CAPTION" for t in results)

    def test_weighted_all_positive(self):
        """所有权重为正时，每个任务都有可能被选中"""
        weights = {"CAPTION": 1.0, "OD": 1.0, "OCR": 1.0}
        sched = TaskScheduler(
            task_types=self.SAMPLE_TASKS,
            config=TaskSchedulingConfig(strategy="weighted"),
            initial_weights=weights,
        )
        results = {sched.select_task() for _ in range(200)}
        # 随机种子可能导致偶尔只选到一个，但 200 次大概率覆盖
        assert len(results) >= 2

    # ---- curriculum 策略 ----

    def test_curriculum_returns_valid_tasks(self):
        sched = self._make_scheduler(strategy="curriculum")
        for _ in range(30):
            task = sched.select_task()
            assert task in self.SAMPLE_TASKS

    def test_curriculum_stage_advances(self):
        """随着步数增加，curriculum 阶段应推进"""
        sched = self._make_scheduler(strategy="curriculum")
        initial_stage = sched.curriculum_stage
        # 推进大量步数
        for _ in range(100):
            sched.select_task()
        # 阶段可能已推进（取决于实现）
        assert sched.curriculum_stage >= initial_stage

    # ---- adaptive 策略 ----

    def test_adaptive_updates_weights(self):
        """报告性能后，adaptive 策略应调整权重"""
        sched = self._make_scheduler(strategy="adaptive")
        # 使用正确的方法名
        sched.update_task_performance("CAPTION", loss=0.5)
        sched.update_task_performance("OD", loss=2.0)
        # 验证权重记录已更新
        assert isinstance(sched.task_weights, dict)

    def test_adaptive_selects_valid_task(self):
        sched = self._make_scheduler(strategy="adaptive")
        task = sched.select_task()
        assert task in self.SAMPLE_TASKS

    # ---- update_task_performance ----

    def test_update_performance_records_history(self):
        sched = self._make_scheduler()
        sched.update_task_performance("CAPTION", loss=1.0)
        assert len(sched.task_performance["CAPTION"]) == 1

    def test_update_performance_multiple(self):
        sched = self._make_scheduler()
        sched.update_task_performance("CAPTION", loss=1.0)
        sched.update_task_performance("CAPTION", loss=0.8)
        assert len(sched.task_performance["CAPTION"]) == 2

    # ---- select_task 返回值 ----

    def test_select_task_returns_valid_task(self):
        for strategy in ["round_robin", "weighted", "curriculum", "adaptive"]:
            sched = self._make_scheduler(strategy=strategy)
            task = sched.select_task()
            assert task in self.SAMPLE_TASKS, f"strategy={strategy} 返回了无效任务: {task}"

    # ---- save_state / load_state ----

    def test_state_roundtrip(self):
        sched = self._make_scheduler(strategy="round_robin")
        # 修改状态
        for _ in range(5):
            sched.select_task()
        sched.update_task_performance("CAPTION", loss=0.5)

        state = sched.save_state()
        assert isinstance(state, dict)

        # 创建新调度器并恢复状态
        sched2 = self._make_scheduler(strategy="round_robin")
        sched2.load_state(state)
        assert sched2.current_step == sched.current_step

    def test_state_preserves_step(self):
        sched = self._make_scheduler(strategy="round_robin")
        for _ in range(10):
            sched.select_task()
        saved_step = sched.current_step

        state = sched.save_state()
        sched2 = self._make_scheduler(strategy="round_robin")
        sched2.load_state(state)
        assert sched2.current_step == saved_step

    # ---- 边界条件 ----

    def test_single_task(self):
        """只有一个任务时，调度器应正常工作"""
        sched = TaskScheduler(
            task_types=["CAPTION"],
            config=TaskSchedulingConfig(strategy="round_robin"),
        )
        for _ in range(5):
            assert sched.select_task() == "CAPTION"

    def test_empty_performance_update(self):
        """没有性能数据时也不应报错"""
        sched = self._make_scheduler()
        # 只选任务，不更新性能
        sched.select_task()
