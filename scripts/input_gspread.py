import time
import gspread

_service_account_filename = "./.config/gspread/beholder_service_account.json"


def input_faq(question: str, answer: str, context: list):
    gs = gspread.service_account(filename=_service_account_filename)

    # FAQ_FeedBack 문서
    document = gs.open_by_key("1iu9H_OZPtnvGGXM07axeCWz8sH5eSSOpi84nrzMX5B0")

    # FeedBack 시트
    worksheet = document.worksheet("feedback")

    # 시트 전체 데이터 가져오기
    # list_of_lists = worksheet.get(return_type=GridRangeType.ListOfLists)

    # 시트 so A열의 데이터 가져오기(특정 열의 데이터가 이미 차있는 경우 사용)
    col_values = worksheet.col_values(1)

    # 업데이트 할 셀 위치
    row_num = len(col_values) + 1
    update_start_cell = f"A{row_num}"
    print(update_start_cell)

    str_context = list_to_string_with_index(context)

    # input 할 데이터
    input_list = []
    input_data = []
    input_data.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    input_data.append(question)
    input_data.append(answer)
    input_data.append(str_context)
    input_list.append(input_data)
    print(input_data)

    # 시트 업데이트
    worksheet.update(range_name=update_start_cell, values=input_list)


def list_to_string_with_index(context):
    # 리스트의 각 요소를 문자열로 변환하고 인덱스를 추가
    result = "".join(
        [f"문서 {i+1}\n{data['page_content']}\n\n" for i, data in enumerate(context)]
    )
    return result
