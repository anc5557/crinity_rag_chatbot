import time
import gspread

_credentials_filename = "./.config/gspread/beholder_credentials.json"
_authorized_user_filename = "./.config/gspread/authorized_user.json"


def input_faq(question: str, answer: str):
    # gs = gspread.oauth()
    gs = gspread.oauth(
        credentials_filename=_credentials_filename,
        authorized_user_filename=_authorized_user_filename,
    )

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

    # input 할 데이터
    input_list = []
    input_data = []
    input_data.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    input_data.append(question)
    input_data.append(answer)
    input_list.append(input_data)
    print(input_data)

    # 시트 업데이트
    worksheet.update(range_name=update_start_cell, values=input_list)
