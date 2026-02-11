from ase import io
from ase.build.general_surface import surface
from ase.visualize import view
import numpy as np

class SlabMaker:
    def __init__(self, 
                 poscar_file,
                 miller, 
                 layers, 
                 super_xyz=[1,1,1], 
                 vacuum=10.0,
                 target_thickness=(14.0, 16.0),
                 max_layers=30):

        self.poscar_file = poscar_file
        self.miller = tuple(miller)
        self.layers = int(layers)
        self.super_xyz = list(super_xyz)
        self.vacuum = float(vacuum)
        self.slab = None

        if not (isinstance(target_thickness, tuple) and len(target_thickness) == 2):
            raise ValueError("target_thickness는 길이 2의 튜플이어야 합니다.")
        self.target_thickness = tuple(target_thickness)

        self.max_layers = int(max_layers)

    def slab_z_fitter(self):
        atoms = io.read(self.poscar_file)
        supercell = atoms.repeat(self.super_xyz)

        layers = self.layers
        slab = surface(lattice=supercell, indices=self.miller, layers=layers)

        # 현재 두께 계산
        z = slab.get_positions()[:, 2]
        thickness = z.max() - z.min()

        # 목표 두께 맞출 때까지 반복
        while thickness < self.target_thickness[0]:

            layers += 1
            if layers > self.max_layers:
                raise RuntimeError("최대 layer 수를 초과했습니다.")

            slab = surface(lattice=supercell, indices=self.miller, layers=layers)
            z = slab.get_positions()[:, 2]
            thickness = z.max() - z.min()

        print(f"✅ 목표두께 도달: {thickness:.2f} Å (layers = {layers})")

        slab.center(vacuum=self.vacuum, axis=2)
        self.slab = slab
        self.layers = layers  # 최종 layer 업데이트
        return self.slab


    def trim_z(self, cutoff=15.0):
        """
        slab 내의 원자들 중 최대 z좌표(z_max)로부터 
        z_max - z_i > cutoff 인 원자를 제거합니다.
        cutoff 기본값은 15 Å.
        """

        if self.slab is None:
            raise ValueError("먼저 slab을 생성해야 합니다. (slab_z_fitter 실행 필요)")

        positions = self.slab.get_positions()
        z_values = positions[:, 2]

        z_max = z_values.max()

        # 삭제할 원자 index 선택 (True=삭제)
        delete_mask = (z_max - z_values) > cutoff

        # 실제 삭제 수행
        self.slab = self.slab[~delete_mask]

        removed = delete_mask.sum()
        print(f"🗑️ 삭제된 원자 수: {removed}개 (cutoff = {cutoff} Å)")

        return self.slab

    def adjust_xy_by_surface_atoms(self,
                                   n_surface: int,
                                   target: int = 16,
                                   max_xy_repeat: int = 6):
        """
        사람이 직접 센 표면 원자 개수(n_surface)를 기준으로
        xy 방향 슈퍼셀을 조정하여 표면 원자 수를 target에 가깝게 맞춥니다.

        - n_surface: 현재 slab에서 사용자가 직접 센 표면 원자 개수
        - target: 맞추고 싶은 표면 원자 수 (기본값 16)
        - max_xy_repeat: super_xyz[0], super_xyz[1]의 최대 반복 수
        """

        if self.slab is None:
            raise ValueError("먼저 slab을 생성해야 합니다. (slab_z_fitter 실행 필요)")

        import numpy as np

        print(f"[XY 조정] 현재 표면 원자 수 = {n_surface}, 목표 = {target}")

        # 1) 이미 목표와 같으면 아무 것도 하지 않음
        if n_surface == target:
            print("✅ 표면 원자 수가 이미 목표와 같습니다. (조정 없음)")
            return self.slab

        # 2) 표면 원자가 부족한 경우 → xy 반복 수 증가 후, 다시 slab 생성 (z fit / trim 포함)
        if n_surface < target:
            if n_surface <= 0:
                scale = 2  # 극단적인 경우 일단 2배
            else:
                # 면적이 원자 수에 비례한다고 보고, 필요한 배수를 근사
                scale = int(np.ceil(np.sqrt(target / n_surface)))

            new_nx = min(self.super_xyz[0] * scale, max_xy_repeat)
            new_ny = min(self.super_xyz[1] * scale, max_xy_repeat)

            if (new_nx == self.super_xyz[0] and
                new_ny == self.super_xyz[1]):
                print("⚠️ xy 반복 수를 더 이상 늘릴 수 없습니다.")
                return self.slab

            print(f"⬆️ 표면 원자 수 부족: super_xyz {self.super_xyz} → [{new_nx}, {new_ny}, {self.super_xyz[2]}]")
            self.super_xyz[0] = new_nx
            self.super_xyz[1] = new_ny

            # bulk 기준으로 다시 slab 생성 (여기서 z fit / z trim 등을 다시 수행하면 됨)
            self.slab_z_fitter()
            # 필요하다면 여기서 바로 self.trim_z(...) 호출 가능
            # self.trim_z(cutoff=15.0)

            return self.slab

        # 3) 표면 원자가 너무 많은 경우 → xy 영역을 줄이고 바깥 원자 삭제
        if n_surface > target:
            atoms = self.slab
            pos = atoms.get_positions()
            x, y = pos[:, 0], pos[:, 1]

            # 현재 xy bounding box
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            width_x = x_max - x_min
            width_y = y_max - y_min

            # target/n_surface 비율만큼 면적 축소 (대략적으로)
            area_ratio = target / n_surface
            scale = np.sqrt(area_ratio)

            new_width_x = width_x * scale
            new_width_y = width_y * scale

            x_center = 0.5 * (x_max + x_min)
            y_center = 0.5 * (y_max + y_min)

            new_x_min = x_center - 0.5 * new_width_x
            new_x_max = x_center + 0.5 * new_width_x
            new_y_min = y_center - 0.5 * new_width_y
            new_y_max = y_center + 0.5 * new_width_y

            # 중앙 사각형 안에 있는 원자만 유지
            keep_xy = ((x >= new_x_min) & (x <= new_x_max) &
                       (y >= new_y_min) & (y <= new_y_max))
            kept = int(keep_xy.sum())
            removed = len(atoms) - kept

            if kept == 0:
                print("⚠️ 잘라낼 영역이 너무 작아 모든 원자가 제거될 위험이 있어 중단합니다.")
                return self.slab

            self.slab = atoms[keep_xy]
            print(f"✂️ xy 영역 축소: 원자 {removed}개 삭제, 남은 원자 {kept}개")

            # 여기서 다시 시각화해서 표면 원자 개수를 사람이 확인하면 됨
            return self.slab



    # ---------------------------------------------------
    #  ⭐ slab 저장 전용 함수
    # ---------------------------------------------------
    def save_poscar(self, filename="POSCAR"):
        if self.slab is None:
            raise ValueError("먼저 slab을 생성해야 합니다. (slab_z_fitter 실행 필요)")
        
        io.write(filename, self.slab, format='vasp')
        print(f"📁 파일 저장 완료: {filename}")

    # ---------------------------------------------------
    def view(self):
        if self.slab is None:
            raise ValueError("먼저 slab을 생성해야 합니다. (slab_z_fitter 실행 필요)")
        view(self.slab)




    def repeat_xy(self, nx: int, ny: int):
        """
        이미 생성된 slab(self.slab)을 xy 방향으로 반복합니다.
        - nx: x 방향 반복 수
        - ny: y 방향 반복 수
        """

        if self.slab is None:
            raise ValueError("먼저 slab을 생성해야 합니다. (slab_z_fitter 실행 필요)")

        if nx <= 0 or ny <= 0:
            raise ValueError("nx와 ny는 1 이상의 정수여야 합니다.")

        # ASE repeat: (nx, ny, nz)
        self.slab = self.slab.repeat((nx, ny, 1))

        # super_xyz 정보도 업데이트 (원래 값에 곱해줄지, 덮어쓸지는 취향인데
        # 여기서는 '덮어쓰기' 대신 '곱하기'로 두었습니다.)
        self.super_xyz[0] *= nx
        self.super_xyz[1] *= ny

        print(f"✅ slab을 xy 방향으로 반복했습니다: repeat = ({nx}, {ny}, 1)")
        print(f"   현재 super_xyz = {self.super_xyz}")

        return self.slab