import numpy as np

# ================= ⚙️ V43: 下划线修正版 =================
# 1. 刚性段 (Prefix="S") -> 生成 SB_0 ... SB_20
PREFIX_1 = "S"
COUNT_1 = 21
LEN_1 = 0.65
POS_1 = [-0.5, 0, 0.6]

# 2. 柔性段 (Prefix="F") -> 生成 FB_0 ... FB_20
PREFIX_2 = "F" 
COUNT_2 = 21
LEN_2 = 0.065
POS_2 = [0.15, 0, 0.6] 

# 3. 其他参数
SLIDER_POS_X = 0.215
START_SLACK_LEN = 0.718
RADIUS_TENDON = 0.002 
TENDON_WIDTH = 0.0003
OUTPUT_XML = "cable_final_v43.xml"

# 物理参数
ROD_BEND = "5000" 
ACTUATOR_KP = "200"
ACTUATOR_KV = "50" 
# ==========================================================

def get_site_str(prefix, r, body_idx):
    # Site 名字：Prefix + S + Index
    name_prefix = f"{prefix}S"
    s = ""
    for k in range(6):
        ang = np.deg2rad(k * 60)
        y, z = r * np.cos(ang), r * np.sin(ang)
        s += f'<site name="{name_prefix}_{body_idx}_{k+1}" pos="0 {y:.5f} {z:.5f}" size="0.00025" rgba="1 0 0 1"/>\n'
    return s

def generate_guide_bodies(prefix, count, length, offset):
    s = ""
    step = length / (count - 1)
    for idx in range(count):
        local_x = idx * step
        abs_pos = np.array(offset) + np.array([local_x, 0, 0])
        guide_name = f"guide_{prefix}_{idx}"
        
        s += f'<body name="{guide_name}" pos="{abs_pos[0]:.4f} {abs_pos[1]:.4f} {abs_pos[2]:.4f}">\n'
        s += '  <joint type="free"/>\n' 
        s += '  <geom class="guide" mass="0.01"/>\n' 
        s += get_site_str(prefix, RADIUS_TENDON, idx)
        s += '</body>\n'
    return s

def assemble_xml():
    # 1. 生成导向体
    guides_1 = generate_guide_bodies(PREFIX_1, COUNT_1, LEN_1, POS_1)
    guides_2 = generate_guide_bodies(PREFIX_2, COUNT_2, LEN_2, POS_2)
    
    # 2. 生成焊接 (🌟 核心修正：加下划线)
    welds_xml = ""
    
    # 刚性段: SB_0 ... SB_20
    for i in range(COUNT_1):
        body_name = f"{PREFIX_1}B_{i}"  # 🌟 注意这里的下划线
        guide_name = f"guide_{PREFIX_1}_{i}"
        welds_xml += f'    <weld name="w_stiff_{i}" body1="{body_name}" body2="{guide_name}"/>\n'
        
    # 柔性段: FB_0 ... FB_20
    for i in range(COUNT_2):
        body_name = f"{PREFIX_2}B_{i}"  # 🌟 注意这里的下划线
        guide_name = f"guide_{PREFIX_2}_{i}"
        welds_xml += f'    <weld name="w_flex_{i}" body1="{body_name}" body2="{guide_name}"/>\n'

    # --- 段间连接 (手动指定首尾) ---
    # 1. 刚性段末端 (SB_20) <-> 柔性段起点 (FB_0)
    # index = count - 1
    rigid_end = f"{PREFIX_1}B_{COUNT_1 - 1}" # SB_20
    flex_start = f"{PREFIX_2}B_0"           # FB_0
    welds_xml += f'    <weld name="connect_rigid_flex" body1="{rigid_end}" body2="{flex_start}" anchor="0 0 0"/>\n'

    # 2. 柔性段末端 (FB_20) <-> 滑块 (slider)
    flex_end = f"{PREFIX_2}B_{COUNT_2 - 1}" # FB_20
    welds_xml += f'    <weld name="connect_flex_slider" body1="{flex_end}" body2="slider" anchor="0 0 0" solimp="0.99 1.0 0.0001" solref="0.0001 1.0"/>\n'

    # 3. 生成肌腱路径
    tendon_xml = ""
    end_site_names = ["s1", "s2", "s3", "s4", "s5", "s6"]
    for k in range(6):
        tendon_xml += f'    <spatial name="tendon{k+1}" width="{TENDON_WIDTH}" rgba="1 0 0 1" limited="true" range="0 0.8">\n'
        tendon_xml += f'      <site site="base_s{k+1}"/>\n' 
        
        # 刚性段 Sites
        p1_name = f"{PREFIX_1}S"
        for idx in range(COUNT_1):
            tendon_xml += f'      <site site="{p1_name}_{idx}_{k+1}"/>\n'
            
        # 柔性段 Sites
        p2_name = f"{PREFIX_2}S"
        for idx in range(COUNT_2):
            tendon_xml += f'      <site site="{p2_name}_{idx}_{k+1}"/>\n'
            
        # 终点
        tendon_xml += f'      <site site="{end_site_names[k]}"/>\n' 
        tendon_xml += f'    </spatial>\n'

    # 4. Actuator
    act_xml = ""
    for k in range(6):
        act_xml += f'    <position name="act_t{k+1}" tendon="tendon{k+1}" kp="{ACTUATOR_KP}" kv="{ACTUATOR_KV}" forcelimited="true" forcerange="-1000 0" ctrlrange="0 0.8"/>\n'

    # 5. Keyframe
    ctrl_vals = "0 0 " + (f"{START_SLACK_LEN:.4f} " * 6)

    # ================= 组装 XML =================
    return f"""
<mujoco model="Cable_Final_V43_UnderscoreFixed">
  <include file="scene.xml"/>
  
  <option timestep="0.001" integrator="implicit" gravity="0 0 0" tolerance="1e-6" impratio="20"/>

  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <compiler autolimits="true"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <keyframe>
    <key name="home" qpos="" ctrl="{ctrl_vals}"/>
  </keyframe>

  <default>
    <default class="guide"><geom type="sphere" size="0.0001" rgba="0 0 0 0" contype="0" conaffinity="0"/></default>
  </default>

  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" pos="0 0 0" size="2 2 .1" type="plane" rgba="0.2 0.2 0.2 1"/>

    <body name="base_plate" pos="-0.5 0 0.6">
      <geom type="cylinder" size="0.004 0.001" rgba=".5 .5 .5 1" euler="0 90 0"/>
      <site name="base_s1" pos="0  0.003 0"        size="0.001" rgba="1 0 0 1"/> 
      <site name="base_s2" pos="0  0.0015 0.002595" size="0.001" rgba="1 1 0 1"/> 
      <site name="base_s3" pos="0 -0.0015 0.002595" size="0.001" rgba="0 1 0 1"/> 
      <site name="base_s4" pos="0 -0.003 0"        size="0.001" rgba="0 1 1 1"/> 
      <site name="base_s5" pos="0 -0.0015 -0.002595" size="0.001" rgba="0 0 1 1"/> 
      <site name="base_s6" pos="0  0.0015 -0.002595" size="0.001" rgba="1 0 1 1"/>
    </body>

    <body name="cable_stiff" pos="{POS_1[0]} {POS_1[1]} {POS_1[2]}">
      <composite prefix="{PREFIX_1}" type="cable" curve="s" count="{COUNT_1} 1 1" size="{LEN_1}" initial="none">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="1e7"/>    
          <config key="bend" value="{ROD_BEND}"/>     
          <config key="vmax" value="1"/>
        </plugin>
        <joint kind="main" damping="2.0" stiffness="0" armature="0.05"/>
        <geom type="cylinder" size=".0036" rgba=".6 .2 .1 0.3" condim="3"/>
      </composite>
    </body>

    <body name="cable" pos="{POS_2[0]} {POS_2[1]} {POS_2[2]}">
      <composite prefix="{PREFIX_2}" type="cable" curve="s" count="{COUNT_2} 1 1" size="{LEN_2}" initial="none">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="1000"/>
          <config key="bend" value="{ROD_BEND}"/>
          <config key="vmax" value=".1"/>
        </plugin>
        <joint kind="main" damping="2.0" stiffness="0" armature="0.05"/>
        <geom type="cylinder" size=".0035" rgba=".8 .2 .1 0.3" condim="3"/>
      </composite>
    </body>
    
    {guides_1}
    {guides_2}

    <body name="slider" pos="{SLIDER_POS_X:.6f} 0 0.6">
      <joint type="free" name="slider_joint"/> 
      <geom size=".001" mass="0.02" rgba="0 0 0 0" contype="0" conaffinity="0"/>
      
      <body name="end_6" pos="0 0 0">
        <geom name="end_6_mass_geom" type="sphere" size="1e-4" mass="0.01" rgba="0 0 0 0" contype="0" conaffinity="0"/>
          <site name="s1" pos="0  0.003 0"        size="0.0005" rgba="1 0 0 1"/> 
          <site name="s2" pos="0  0.0015 0.002595" size="0.0005" rgba="1 1 0 1"/> 
          <site name="s3" pos="0 -0.0015 0.002595" size="0.0005" rgba="0 1 0 1"/> 
          <site name="s4" pos="0 -0.003 0"        size="0.0005" rgba="0 1 1 1"/> 
          <site name="s5" pos="0 -0.0015 -0.002595" size="0.0005" rgba="0 0 1 1"/> 
          <site name="s6" pos="0  0.0015 -0.002595" size="0.0005" rgba="1 0 1 1"/>
      </body>
    </body>
  </worldbody>

  <equality>
    {welds_xml}
  </equality>

  <contact>
    <exclude body1="{PREFIX_2}B_{COUNT_2-1}" body2="slider" />
  </contact>

  <tendon>
    {tendon_xml}
  </tendon>

  <actuator>
    <motor site="S_last" gear="0 0 1 0 0 0" ctrlrange="-100 100"/>
    <motor site="S_last" gear="0 1 0 0 0 0" ctrlrange="-100 100"/>
    {act_xml}
  </actuator>

</mujoco>
"""

def main():
    final_xml = assemble_xml()
    with open(OUTPUT_XML, "w", encoding="utf-8") as f:
        f.write(final_xml)
    print(f"🎉 V43 下划线修正版生成: {OUTPUT_XML}")
    print("   ✅ 命名规则已修正为: [Prefix] + 'B_' + [Index] (例如: SB_0, FB_20)")
    print("   ✅ 这次完全符合 MuJoCo 的默认索引命名规则。")

if __name__ == "__main__":
    main()