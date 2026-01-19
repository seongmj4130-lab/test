# -*- coding: utf-8 -*-
"""
모든 boolean 설정을 true로 변경하여 재현 가능하게 만듦
"""

import yaml
import re

def enable_all_features():
    """모든 boolean 설정을 true로 변경"""

    # config.yaml 읽기
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        content = f.read()

    # 정규식으로 false를 true로 변경 (단독으로 있는 false만)
    # (?<!-)false(?!-) : 앞뒤에 -가 없는 false만 매칭
    pattern = r'(?<!-)\bfalse\b(?!-)'
    content = re.sub(pattern, 'true', content)

    # 설정 저장
    with open('configs/config.yaml', 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 모든 boolean 설정을 true로 변경 완료")
    print("재현 가능성을 위해 모든 고급 기능이 활성화되었습니다.")

    # 변경된 내용 확인
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        final_content = f.read()

    false_count = len(re.findall(r'\bfalse\b', final_content))
    true_count = len(re.findall(r'\btrue\b', final_content))

    print(f"변경 결과: false → true (총 {true_count}개 true 설정 활성화)")

if __name__ == "__main__":
    enable_all_features()