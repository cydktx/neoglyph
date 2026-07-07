from neoglyph import Tensor, NeoGlyphVM


def main():
    vm = NeoGlyphVM(verbose=True)
    script = """
    // 计算 d = (a+b)*b，验证梯度
    PUSH 2
    STORE a
    PUSH 3
    STORE b

    TAPE
    LOAD a
    LOAD b
    ADD
    LOAD b
    MUL
    STORE d
    UNTAPE

    LOAD d
    GRAD

    LOAD a
    PRINT
    LOAD b
    PRINT
    HALT
    """
    print("运行示例脚本: d = (a+b) * b")
    print("输入: a=2, b=3")
    print("期望输出: d=15")
    print("梯度种子: d=15")
    print("期望梯度: da=3*15=45, db=8*15=120")
    print("-" * 40)
    vm.run(script)


if __name__ == "__main__":
    main()
