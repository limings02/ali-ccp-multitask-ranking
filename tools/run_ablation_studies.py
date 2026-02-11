#!/usr/bin/env python3
"""
python tools/run_ablation_studies.py
"""
"""
运行消融实验脚本

依次运行四个实验配置：
1. mmoe_optim/base_mmoe.yaml - 基础 MMoE + Gate Stabilize (低正则权重)
2. mmoe_optim/midgate_mmoe.yaml - MMoE + Gate Stabilize (高正则权重)
3. ple_optim/hetero_ple.yaml - PLE 带 shared_only gate 正则
4. ple_optim/gateall_hetero_ple.yaml - PLE 带 all gate 正则
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

# 实验配置列表
EXPERIMENTS = [
    {
    "name": "classic_mmoe",
    "config": "configs/experiments/mmoe_optim/test1_mmoe.yaml",
    "description": "mmoe)"
    },
    {
        "name": "focal_mmoe",
        "config": "configs/experiments/mmoe_optim/test2_mmoe.yaml",
        "description": "PLE homo)"
    },
    {
        "name": "classic_ple",
        "config": "configs/experiments/ple_optim/base_homo_ple.yaml",
        "description": "PLE hetero"
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
            logger.info(f"✓ 实验 {exp_name} 运行成功")
            return True
        else:
            logger.error(f"✗ 实验 {exp_name} 运行失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"✗ 实验 {exp_name} 执行异常: {e}")
        return False


def main():
    """主函数"""
    logger.info(f"开始运行消融实验集合，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"总计 {len(EXPERIMENTS)} 个实验")
    
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
    logger.info("消融实验运行总结")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for exp_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        logger.info(f"{status}: {exp_name}")
    
    logger.info(f"\n总计: {passed} 个成功，{failed} 个失败")
    logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
