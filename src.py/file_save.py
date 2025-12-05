import os

def save_filtered_results(filter_instance, text_list, output_folder="filtered_results", filename="cleaned_log.txt"):
    """
    텍스트 리스트를 필터링하여 지정된 폴더의 파일로 저장하는 함수
    (줄 단위로 쪼개지 않고 텍스트 전체를 하나로 처리)
    """
    
    # 1. 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"폴더 생성 완료: {output_folder}")
    
    # 2. 파일 전체 경로 설정
    file_path = os.path.join(output_folder, filename)

    # 3. 파일 쓰기
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== 욕설 필터링 결과 로그 (통문장 처리) ===\n\n")
        
        for idx, text in enumerate(text_list, 1):
            text = text.strip()
            if not text: continue

            f.write(f"[Example {idx}]\n")
            
            # 전체 텍스트를 한 번에 처리
            # 주의: 현재 _normalize 로직에 의해 줄바꿈이 공백으로 바뀔 수 있습니다.
            cleaned_text = filter_instance.clean(text)
            is_bad = filter_instance.has_profanity(text)
            
            # 파일에 기록할 포맷
            f.write(f"원문:\n{text}\n\n")
            f.write(f"정제 결과:\n{cleaned_text}\n\n")
            f.write(f"욕 포함?: {'True' if is_bad else '정상'}\n")
            f.write("\n" + "="*50 + "\n\n")

    print(f"저장 완료 파일경로: {file_path}")
