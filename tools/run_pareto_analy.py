#!/usr/bin/env python3
"""
批量运行 Pareto 分析实验脚本

使用方法:
    python tools/run_pareto_analy.py

说明:
    依次运行 configs/experiments/mmoe_optim/pareto_analy/ 下的所有实验配置
    测试不同 lambda 权重对模型性能的影响
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# Pareto 分析实验配置目录
PARETO_ANALY_DIR = PROJECT_ROOT / "configs/experiments/mmoe_optim/pareto_analy"

# 实验配置列表
EXPERIMENTS = [
    {
        "name": "classic_mmoe_lambda1",
        "config": "configs/experiments/mmoe_optim/pareto_analy/classic_mmoe_lambda1.yaml",
        "description": "Classic MMoE with lambda_ctcvr=1.0"
    },
    {
        "name": "classic_mmoe_lambda3",
        "config": "configs/experiments/mmoe_optim/pareto_analy/classic_mmoe_lambda3.yaml",
        "description": "Classic MMoE with lambda_ctcvr=3.0"
    },
    {
        "name": "classic_mmoe_lambda5",
        "config": "configs/experiments/mmoe_optim/pareto_analy/classic_mmoe_lambda5.yaml",
        "description": "Classic MMoE with lambda_ctcvr=5.0"
    },
    {
        "name": "classic_mmoe_lambda7",
        "config": "configs/experiments/mmoe_optim/pareto_analy/classic_mmoe_lambda7.yaml",
        "description": "Classic MMoE with lambda_ctcvr=7.0"
    },
    {
        "name": "classic_mmoe_lambda10",
        "config": "configs/experiments/mmoe_optim/pareto_analy/classic_mmoe_lambda10.yaml",
        "description": "Classic MMoE with lambda_ctcvr=10.0"
    },
    {
        "name": "classic_mmoe_lambda13",
        "config": "configs/experiments/mmoe_optim/pareto_analy/classic_mmoe_lambda13.yaml",
        "description": "Classic MMoE with lambda_ctcvr=13.0"
    },
    {
        "name": "use_resi_mmoe_lambda5",
        "config": "configs/experiments/mmoe_optim/pareto_analy/use_resi_mmoe_lambda5.yaml",
        "description": "MMoE with residual enabled, lambda_ctcvr=5.0"
    },
    {
        "name": "use_focal_mmoe_lambda5",
        "config": "configs/experiments/mmoe_optim/pareto_analy/use_focal_mmoe_lambda5.yaml",
        "description": "MMoE with aux focal enabled, lambda_ctcvr=5.0"
    },
    {
        "name": "use_focal_resi_mmoe_lambda5",
        "config": "configs/experiments/mmoe_optim/pareto_analy/use_focal_resi_mmoe_lambda5.yaml",
        "description": "MMoE with residual + aux focal, lambda_ctcvr=5.0"
    },
]


def run_experiment(config_path: str, exp_name: str, exp_desc: str) -> bool:
    """
    运行单个实验
    
    Args:
        config_path: 相对于项目根目录的配置文件路径
        exp_name: 实验名称
        exp_desc: 实验描述
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"开始运行实验: {exp_name}")
    logger.info(f"描述: {exp_desc}")
    logger.info(f"配置: {config_path}")
    logger.info("=" * 80)
    
    config_full_path = PROJECT_ROOT / config_path
    
    # 检查配置文件是否存在
    if not config_full_path.exists():
        logger.error(f"配置文件不存在: {config_full_path}")
        return False
    
    # 构建命令
    cmd = [
        sys.executable,
        "-m",
        "src.cli.main",
        "train",
        "--config",
        str(config_full_path)
    ]
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行实验
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,  # 不在返回码非0时抛异常
            capture_output=False  # 显示输出
        )
        
        if result.returncode == 0:
            logger.info(f"[OK] 实验 {exp_name} 运行成功")
            return True
        else:
            logger.error(f"[FAIL] 实验 {exp_name} 运行失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"[FAIL] 实验 {exp_name} 执行异常: {e}")
        return False


def main():
    """主函数"""
    logger.info(f"开始运行 Pareto 分析实验集合，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"实验配置目录: {PARETO_ANALY_DIR}")
    logger.info(f"总计 {len(EXPERIMENTS)} 个实验")
    
    # 检查实验目录是否存在
    if not PARETO_ANALY_DIR.exists():
        logger.error(f"实验配置目录不存在: {PARETO_ANALY_DIR}")
        return 1
    
    results = {}
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        logger.info(f"\n[{i}/{len(EXPERIMENTS)}] 运行实验...")
        success = run_experiment(
            exp["config"],
            exp["name"],
            exp["description"]
        )
        results[exp["name"]] = success
        
        if not success:
            logger.warning(f"实验 {exp['name']} 失败，继续下一个实验...")
    
    # 输出总结
    logger.info("\n" + "=" * 80)
    logger.info("Pareto 分析实验运行总结")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for exp_name, success in results.items():
        status = "[OK] 成功" if success else "[FAIL] 失败"
        logger.info(f"{status}: {exp_name}")
    
    logger.info(f"\n总计: {passed} 个成功，{failed} 个失败")
    logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
