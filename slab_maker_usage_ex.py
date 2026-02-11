# run_slab.py
from slabmaker import SlabMaker

# 1) SlabMaker 생성
maker = SlabMaker(
    poscar_file="/Users/seungsoohan/chun_lab/opencatal/slab_test/POSCAR_Pd_slab",   # 실제 파일 이름/경로에 맞게 수정
    miller=[7, 1, 1],
    layers=6,
    super_xyz=[1, 1, 1],
    vacuum=12.0,
    target_thickness=(20.0, 22.0),
    max_layers=100
)

# 2) slab 생성 (z 방향 두께 맞추기)
maker.slab_z_fitter()


# 3) 시각화 → 여기서 표면 원자 개수 직접 세기
maker.view()


# 4) xy 반복수 설정 (사용자로부터 반복수 입력받아서 현재 slab을 바로 반복)
while True:
    try:
        repeat_input = input("xy 반복수를 입력하시오 (예: '2 3' → x=2, y=3): ")
        nx, ny = map(int, repeat_input.split())

        if nx <= 0 or ny <= 0:
            print("⚠️ 반복수는 1 이상의 정수여야 합니다.")
            continue

        break

    except ValueError:
        print("⚠️ 공백으로 구분된 두 개의 정수를 입력해야 합니다. 예: 2 3")



# 이미 만들어진 slab을 그대로 반복
maker.repeat_xy(nx, ny)
# 5) z 방향으로 다시 trim (밑에 튀어나온 원자 정리)
maker.trim_z(cutoff=15.0)

# 6) 시각화 → 여기서 표면 원자 개수 직접 세기
maker.view()

# 7) 최종 slab 저장
#maker.save_poscar("POSCAR_111_xy_ztrim")
