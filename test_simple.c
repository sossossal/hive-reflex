// 最简化的 RISC-V 测试程序
// 无需复杂的库，直接测试工具链

void _start(void) {
  // 简单的死循环
  volatile int x = 0;
  while (1) {
    x++;
  }
}
