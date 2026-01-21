#!/usr/bin/env python3
"""
权重烧录工具 - 将编译后的模型权重打包并烧录到 FLASH

使用方法:
    python flash_model.py --weights model_weights.bin --output model.flash
    
然后使用硬件烧录工具将 model.flash 烧录到指定地址
"""

import argparse
import struct
import hashlib
import zlib
from pathlib import Path

# 模型格式定义
MODEL_MAGIC = 0x43494D32  # "CIM2"
MODEL_VERSION = 0x0200    # v2.0

class ModelPacker:
    """模型打包器"""
    
    def __init__(self):
        self.header = {}
        self.config = {}
        self.weights = b''
        
    def pack(self, weights_path, config_dict, output_path, model_name="ReflexNet"):
        """
        打包模型
        
        Args:
            weights_path: 权重文件路径
            config_dict: 配置字典
            output_path: 输出文件路径
            model_name: 模型名称
        """
        print(f"📦 打包模型: {model_name}")
        
        # 读取权重
        with open(weights_path, 'rb') as f:
            self.weights = f.read()
        
        print(f"  权重大小: {len(self.weights)} bytes")
        
        # 构建配置
        self.config = config_dict
        config_bytes = self._encode_config(config_dict)
        
        # 计算偏移
        header_size = 128  # 固定 128 字节头
        config_offset = header_size
        weight_offset = config_offset + len(config_bytes)
        
        # 计算哈希
        model_hash = hashlib.sha256(self.weights).hexdigest()
        
        # 构建头
        header_size_actual = 128
        model_size = header_size + len(config_bytes) + len(self.weights)
        
        # 打包头 (128 字节)
        header = struct.pack(
            '<IHHIIIII32s64s',
            MODEL_MAGIC,                    # magic (4)
            MODEL_VERSION,                  # version (2)
            0,                              # reserved (2)
            model_size,                     # model_size (4)
            weight_offset,                  # weight_offset (4)
            len(self.weights),              # weight_size (4)
            config_offset,                  # config_offset (4)
            len(config_bytes),              # config_size (4)
            model_name.encode('utf-8')[:32],  # model_name (32)
            model_hash.encode('utf-8')[:64]   # model_hash (64)
        )
        
        # 头部填充到 128 字节
        header += b'\x00' * (header_size_actual - len(header) - 4)
        
        # 计算 CRC (排除 CRC 字段)
        crc32 = zlib.crc32(header)
        header += struct.pack('<I', crc32)
        
        # 组合所有部分
        model_data = header + config_bytes + self.weights
        
        # 写入文件
        with open(output_path, 'wb') as f:
            f.write(model_data)
        
        print(f"✓ 模型已打包: {output_path}")
        print(f"  总大小: {len(model_data)} bytes")
        print(f"  魔数: 0x{MODEL_MAGIC:08X}")
        print(f"  版本: 0x{MODEL_VERSION:04X}")
        print(f"  CRC32: 0x{crc32:08X}")
        print(f"  SHA256: {model_hash[:16]}...")
        
        return output_path
        
    def _encode_config(self, config):
        """编码配置为二进制"""
        # 配置结构 (32 字节)
        config_bytes = struct.pack(
            '<IIIIIffi',
            config.get('input_size', 12),
            config.get('output_size', 1),
            config.get('hidden_size', 16),
            config.get('num_layers', 3),
            config.get('dtype', 0),  # 0=INT8, 2=FP32
            config.get('has_lstm', 1),
            config.get('quant_scale', 1.0),
            config.get('quant_zero', 0)
        )
        return config_bytes


def generate_flash_script(model_path, flash_addr, output_script):
    """
    生成烧录脚本
    
    Args:
        model_path: 模型文件路径
        flash_addr: FLASH 目标地址
        output_script: 输出脚本路径
    """
    # OpenOCD 烧录脚本
    script_content = f"""# OpenOCD Flash 脚本
# 自动生成

# 初始化
init
reset halt

# 擦除 Flash
flash erase_address 0x{flash_addr:08X} 0x100000

# 烧录模型
flash write_image {model_path} 0x{flash_addr:08X}

# 验证
verify_image {model_path} 0x{flash_addr:08X}

# 复位并运行
reset run

# 退出
shutdown
"""
    
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    print(f"\n✓ OpenOCD 脚本已生成: {output_script}")
    print(f"\n烧录命令:")
    print(f"  openocd -f interface/jlink.cfg -f target/riscv.cfg -f {output_script}")


def main():
    parser = argparse.ArgumentParser(description='模型权重烧录工具')
    parser.add_argument('--weights', required=True, help='权重文件路径')
    parser.add_argument('--output', default='model.flash', help='输出文件路径')
    parser.add_argument('--name', default='ReflexNet', help='模型名称')
    parser.add_argument('--input-size', type=int, default=12, help='输入维度')
    parser.add_argument('--output-size', type=int, default=1, help='输出维度')
    parser.add_argument('--hidden-size', type=int, default=16, help='隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=3, help='层数')
    parser.add_argument('--dtype', choices=['int8', 'fp32'], default='fp32', help='数据类型')
    parser.add_argument('--has-lstm', action='store_true', default=True, help='包含 LSTM')
    parser.add_argument('--quant-scale', type=float, default=1.0, help='量化缩放')
    parser.add_argument('--flash-addr', default='0x08090000', help='FLASH 地址')
    parser.add_argument('--gen-script', action='store_true', help='生成烧录脚本')
    
    args = parser.parse_args()
    
    # 配置字典
    config = {
        'input_size': args.input_size,
        'output_size': args.output_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dtype': 0 if args.dtype == 'int8' else 2,
        'has_lstm': 1 if args.has_lstm else 0,
        'quant_scale': args.quant_scale,
        'quant_zero': 0
    }
    
    # 打包模型
    packer = ModelPacker()
    output_path = packer.pack(args.weights, config, args.output, args.name)
    
    # 生成烧录脚本
    if args.gen_script:
        flash_addr = int(args.flash_addr, 16)
        script_path = args.output.replace('.flash', '.ocd')
        generate_flash_script(output_path, flash_addr, script_path)
    
    print("\n✅ 完成!")
    print("\n下一步:")
    print("  1. 检查生成的 .flash 文件")
    print("  2. 使用 OpenOCD 或 J-Link 烧录到芯片")
    print("  3. 复位芯片，模型将自动从 FLASH 加载")


if __name__ == '__main__':
    main()
