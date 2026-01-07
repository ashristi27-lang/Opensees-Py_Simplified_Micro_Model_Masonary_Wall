"""
Created on Wed Dec  24 12:28:59 2025

@author: Shristi Aryal
"""

import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 2)
# Standard Nepali Brick Size approx 230x110x55 or 210x105x55
brick_L = 210.0
brick_H = 55.0
mortar_t = 10.0
wall_thick = 230.0

dx_unit = brick_L + mortar_t
dy_unit = brick_H + mortar_t
dx_mesh = dx_unit / 2.0   
dy_mesh = dy_unit

N_cols = 9
N_rows = 27
# Values from Gautam et al. and Mishra et al. for Nepali Masonry
brick_E = 4300.0
brick_nu = 0.24
brick_rho = 1900.0 

mortar_E = 150.0   
mortar_nu = 0.2
G_mortar = mortar_E / (2 * (1 + mortar_nu))

Kn = mortar_E / mortar_t  
Ks = G_mortar / mortar_t  

cohesion = 0.02 
friction_angle = 30.0
mu = np.tan(np.radians(friction_angle))

ops.nDMaterial('ElasticIsotropic', 1, brick_E, brick_nu)

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

brick_elems = []
node_list = []

for r in range(N_rows):
    is_odd = (r % 2 != 0)
    x_start = 0.5 * dx_unit if is_odd else 0.0
    cols = N_cols if not is_odd else N_cols - 1
    
    for c in range(cols):
        x = x_start + c * dx_unit
        y = r * dy_unit
        
        tag_base = int(100000*(r+1) + 1000*(c+1))
        nds = [tag_base + i for i in range(1, 7)]
        
        pts = [(x, y), (x+dx_mesh, y), (x+2*dx_mesh, y),
               (x, y+dy_mesh), (x+dx_mesh, y+dy_mesh), (x+2*dx_mesh, y+dy_mesh)]
        
        for i, (nx, ny) in enumerate(pts):
            ops.node(nds[i], nx, ny)
            node_list.append(nds[i])
            if r == 0 and ny <= 0.01:
                ops.fix(nds[i], 1, 1) 

        ops.element('quad', tag_base+10, nds[0], nds[1], nds[4], nds[3], wall_thick, 'PlaneStress', 1)
        ops.element('quad', tag_base+11, nds[1], nds[2], nds[5], nds[4], wall_thick, 'PlaneStress', 1)
        brick_elems.extend([tag_base+10, tag_base+11])
        
        mass = (dx_unit/2.0 * dy_unit * wall_thick) * (brick_rho * 9.81 / 1e9)
        
        for n in [nds[0], nds[1], nds[4], nds[3]]: ops.load(n, 0.0, -mass/4.0)
        for n in [nds[1], nds[2], nds[5], nds[4]]: ops.load(n, 0.0, -mass/4.0)

spring_tag = 4000000
nodes_by_pos = {}
for t in ops.getNodeTags():
    x, y = ops.nodeCoord(t)
    k = f"{x:.3f}_{y:.3f}"
    if k not in nodes_by_pos: nodes_by_pos[k] = []
    nodes_by_pos[k].append(t)

for k, tags in nodes_by_pos.items():
    if len(tags) < 2: continue
    tags.sort()
    t1, t2 = tags[0], tags[1]
    
   
    y_loc = ops.nodeCoord(t1)[1]
    depth = (N_rows * dy_unit) - y_loc
    sigma_n = max(brick_rho * 9.81 / 1e9 * depth, 0.01) 
    

    is_bed = ((t1 // 100000) != (t2 // 100000))
    area = (dx_unit * wall_thick)/2.0 if is_bed else (dy_unit * wall_thick)
    
    
    F_yield = (cohesion + mu * sigma_n) * area
    K_s = Ks * area
    K_n = Kn * area
    
    ops.uniaxialMaterial('Steel01', spring_tag+1, F_yield, K_s, 0.001)
    ops.uniaxialMaterial('ENT', spring_tag+2, K_n)
    

    d1, d2 = (1, 2) if is_bed else (2, 1)
    ops.element('zeroLength', spring_tag, t1, t2, '-mat', spring_tag+1, spring_tag+2, '-dir', d1, d2)
    spring_tag += 3

print("Applying Gravity...")
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-5, 20)
ops.algorithm('Newton')
ops.integrator('LoadControl', 0.1)
ops.analysis('Static')
ops.analyze(10)
ops.loadConst('-time', 0.0)

print("Running Pushover...")
ctrl_node = [t for t in ops.getNodeTags() if ops.nodeCoord(t)[1] > (N_rows-1)*dy_unit][0]
wall_height_mm = N_rows * dy_unit 

ops.timeSeries('Linear', 2)
ops.pattern('Plain', 2, 2)
ops.load(ctrl_node, 1.0, 0.0)

disp_data = [0.0]
shear_data = [0.0]

target_snapshots = [0.5, 3.0, 9.5] # mm 
captured_flags = [False, False, False] 
saved_shapes = [] 

initial_shape = {t: ops.nodeDisp(t) for t in ops.getNodeTags()}
saved_shapes.append(initial_shape)

max_disp = 10.0 # mm
du = 0.05       
current_disp = 0.0

ops.integrator('DisplacementControl', ctrl_node, 1, du)
ops.analysis('Static')

for i in range(1000): 
    ok = ops.analyze(1)
    
    if ok != 0:
        print(f"Structure Failed at Step {i}")
        break 
    
    d = ops.nodeDisp(ctrl_node, 1)
    
    ops.reactions()
    r_base = 0.0
    for t in ops.getNodeTags():
        if ops.nodeCoord(t)[1] < 1.0: r_base += ops.nodeReaction(t, 1)
            
    disp_data.append(d)
    shear_data.append(-r_base)
    
    for idx, target in enumerate(target_snapshots):
        if not captured_flags[idx] and abs(d) >= target:
            print(f"--> Capturing Snapshot at {d:.2f} mm")
            
            current_shape = {t: ops.nodeDisp(t) for t in ops.getNodeTags()}
            saved_shapes.append(current_shape)
            captured_flags[idx] = True
            
    if abs(d) >= max_disp: break

final_shape = {t: ops.nodeDisp(t) for t in ops.getNodeTags()}
if len(saved_shapes) < 4: saved_shapes.append(final_shape)

print("Analysis Done.")

if len(disp_data) > 1:
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3) 
    
    ax_curve = fig.add_subplot(gs[:, :2]) 
    
    ax_snap1 = fig.add_subplot(gs[0, 2])
    ax_snap2 = fig.add_subplot(gs[1, 2])
    ax_snap3 = fig.add_subplot(gs[2, 2])
    snap_axes = [ax_snap1, ax_snap2, ax_snap3]

    base_shear_kN = np.array(shear_data) / 1000.0
    ax_curve.plot(disp_data, base_shear_kN, color='#D32F2F', linewidth=3, label='Response')
    
    ax_curve.axvspan(0, 1.0, color='green', alpha=0.1, label='Immediate Occupancy (IO)')
    ax_curve.axvspan(1.0, 6.0, color='orange', alpha=0.1, label='Life Safety (LS)')
    ax_curve.axvspan(6.0, 10.5, color='red', alpha=0.1, label='Collapse Prevention (CP)')

    ax_curve.set_title('Mud Mortar Wall: Seismic Capacity', fontsize=16, fontweight='bold')
    ax_curve.set_xlabel('Roof Displacement (mm)', fontsize=14)
    ax_curve.set_ylabel('Base Shear (kN)', fontsize=14)
    ax_curve.grid(True, linestyle='--', alpha=0.6)
    ax_curve.legend(loc='lower right', fontsize=12)

    ax_drift = ax_curve.twiny()
    ax_drift.set_xlim(ax_curve.get_xlim())
    new_tick_locs = ax_curve.get_xticks()
    def tick_function(X):
        return ["%.2f%%" % (z / wall_height_mm * 100) for z in X]
    ax_drift.set_xticklabels(tick_function(new_tick_locs))
    ax_drift.set_xlabel('Drift Ratio (%)', fontsize=12, color='gray')

    def plot_wall_shape(ax, shape_data, title, scale=12):

        ax.set_aspect('equal')
        ax.axis('off') 
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        for ele in brick_elems:
            nds = ops.eleNodes(ele)
            
            poly_coords = []
            for n in nds:
                c = ops.nodeCoord(n)
                d = shape_data[n] 
                
                x_new = c[0] + d[0] * scale
                y_new = c[1] + d[1] * scale
                poly_coords.append([x_new, y_new])
            
            rect = Polygon(poly_coords, closed=True, facecolor='#B22222', edgecolor='black', linewidth=0.8)
            ax.add_patch(rect)
        
        ax.autoscale_view()

    titles = [
        f"A. Yield Point (~{target_snapshots[0]}mm)", 
        f"B. Plastic Hinge (~{target_snapshots[1]}mm)", 
        f"C. Ultimate Limit (~{target_snapshots[2]}mm)"
    ]
    
    shapes_to_plot = saved_shapes[1:] 
    
    for i, ax in enumerate(snap_axes):
        if i < len(shapes_to_plot):
            
            plot_wall_shape(ax, shapes_to_plot[i], titles[i], scale=20) 
            
            idx = (np.abs(np.array(disp_data) - target_snapshots[i])).argmin()
            ax_curve.plot(disp_data[idx], base_shear_kN[idx], 'ko', markersize=10)
            ax_curve.annotate(['A', 'B', 'C'][i], 
                              (disp_data[idx], base_shear_kN[idx]), 
                              xytext=(0, 12), textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Seismic_Capacity_Burntbrick_Mudmortar.png', dpi=300)
    print("Seismic_Capacity_Burntbrick_Mudmortar.png")
    plt.show()

else:
    print("No data to plot.")