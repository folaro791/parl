import time
import sys

def advanced_blinking():
    """高级闪烁效果"""
    # ANSI 转义序列
    BLINK_SLOW = '\033[5m'   # 慢速闪烁
    BLINK_FAST = '\033[6m'   # 快速闪烁（部分终端支持）
    REVERSE = '\033[7m'      # 反色
    RESET = '\033[0m'
    
    colors = {
        'red': '\033[91m',
        'green': '\033[92m', 
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    
    messages = [
        "🚨 系统警告",
        "⚠️ 注意安全", 
        "🔴 紧急状态",
        "💀 危险操作"
    ]
    
    for i in range(30):
        color_idx = i % len(colors)
        color = list(colors.values())[color_idx]
        message = messages[i % len(messages)]
        
        if i % 3 == 0:
            # 慢速闪烁
            output = f"{BLINK_SLOW}{color}{message}{RESET}"
        elif i % 3 == 1:
            # 反色闪烁
            output = f"{BLINK_SLOW}{REVERSE}{message}{RESET}"
        else:
            # 快速闪烁（如果支持）
            output = f"{BLINK_FAST}{color}{message}{RESET}"
        
        sys.stdout.write(f"\r{output} 计数: {i+1}")
        sys.stdout.flush()
        time.sleep(0.3)

advanced_blinking()