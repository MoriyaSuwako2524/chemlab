#!/usr/bin/env python3
"""
诊断脚本：检查MECP优化中的梯度shape和约束力问题
"""
import numpy as np
from chemlab.util.mecp import mecp, mecp_soc
from chemlab.util.file_system import qchem_file


def check_gradient_shapes():
    """检查梯度和结构的shape是否一致"""
    print("=" * 60)
    print("检查梯度和结构的shape")
    print("=" * 60)

    # 创建一个测试MECP对象
    test_mecp = mecp_soc()
    test_mecp.ref_path = "./"
    test_mecp.ref_filename = "test.in"  # 需要一个测试文件

    try:
        test_mecp.read_init_structure()

        natom = test_mecp.state_1.inp.molecule.natom
        print(f"原子数: {natom}")

        # 检查return_xyz_list()的shape
        structure = test_mecp.state_1.inp.molecule.return_xyz_list().astype(float)
        print(f"return_xyz_list() shape: {structure.shape}")
        print(f"期望: ({natom}, 3)")

        # 检查flatten后的shape
        flattened = structure.flatten()
        print(f"flatten后 shape: {flattened.shape}")
        print(f"期望: ({3 * natom},)")

        # 检查reshape后的shape
        reshaped = flattened.reshape((natom, 3))
        print(f"reshape((natom, 3)) shape: {reshaped.shape}")

        # 测试约束力
        if natom >= 2:
            print(f"\n测试约束力计算 (原子0-1, 目标2.0 Å, K=100):")
            test_mecp.add_restrain(0, 1, 2.0, 100.0)

            current_dist = test_mecp.state_1.inp.molecule.calc_distance_of_2_atoms(0, 1)
            print(f"当前距离: {current_dist:.4f} Å")

            F = test_mecp.restrain_force(0, 1, 2.0, 100.0)
            print(f"约束力 shape: {F.shape}")
            print(f"期望: (3, {natom})")
            print(f"约束力范数: {np.linalg.norm(F):.6f}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def compare_mecp_classes():
    """比较mecp和mecp_soc的区别"""
    print("\n" + "=" * 60)
    print("比较mecp和mecp_soc类")
    print("=" * 60)

    m1 = mecp()
    m2 = mecp_soc()

    print(f"mecp.different_type: {m1.different_type}")
    print(f"mecp_soc.different_type: {m2.different_type}")

    print(f"\nmecp方法:")
    print(f"  - check_convergence: {hasattr(m1, 'check_convergence')}")

    print(f"\nmecp_soc方法:")
    print(f"  - check_converge: {hasattr(m2, 'check_converge')}")
    print(f"  - check_convergence: {hasattr(m2, 'check_convergence')}")


def test_constraint_logic():
    """测试约束力添加逻辑"""
    print("\n" + "=" * 60)
    print("测试约束力添加逻辑")
    print("=" * 60)

    # 场景1: 使用add_restrain
    print("\n场景1: 使用add_restrain()")
    m = mecp()
    m.add_restrain(0, 1, 2.0, 100.0)
    print(f"  restrain = {m.restrain}")
    print(f"  restrain_list = {m.restrain_list}")
    print(f"  在calc_new_gradient()中会自动添加约束力")

    # 场景2: 不使用add_restrain
    print("\n场景2: 不使用add_restrain()")
    m = mecp()
    print(f"  restrain = {m.restrain}")
    print(f"  restrain_list = {m.restrain_list}")
    print(f"  需要手动添加约束力")


def check_shape_consistency():
    """检查mecp_soc的shape一致性问题"""
    print("\n" + "=" * 60)
    print("检查mecp_soc的shape一致性")
    print("=" * 60)

    print("\n在mecp基类中:")
    print("  update_structure():")
    print("    structure = return_xyz_list()  # (natom, 3)")
    print("    self.last_structure = x_k.reshape((natom, 3))")
    print("\n  check_convergence():")
    print("    current = return_xyz_list()  # (natom, 3)")
    print("    last = self.last_structure  # (natom, 3) - 一致!")

    print("\n在mecp_soc类中:")
    print("  update_structure(): 继承自基类")
    print("    self.last_structure.shape = (natom, 3)")
    print("\n  check_converge(): 重写")
    print("    current = return_xyz_list().T  # (3, natom)")
    print("    last = self.last_structure.reshape((3, natom))  # ❌ 不一致!")
    print("\n⚠️  这会导致shape不匹配，位移计算错误!")


if __name__ == "__main__":
    print("MECP 优化诊断工具\n")

    # 1. 比较类
    compare_mecp_classes()

    # 2. 测试约束逻辑
    test_constraint_logic()

    # 3. 检查shape一致性
    check_shape_consistency()

    # 4. 如果有测试文件，检查梯度shape
    import os

    if os.path.exists("ref.in") or os.path.exists("test.in"):
        check_gradient_shapes()
    else:
        print("\n" + "=" * 60)
        print("⚠️  未找到测试文件 (ref.in 或 test.in)")
        print("请提供一个Q-Chem输入文件以测试梯度shape")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("诊断完成!")
    print("=" * 60)
    print("\n建议:")
    print("1. 使用add_restrain()而不是手动添加约束力")
    print("2. 修复mecp_soc.check_converge()中的shape不一致问题")
    print("3. 确保梯度和结构的shape匹配")