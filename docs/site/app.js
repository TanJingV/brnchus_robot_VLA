/* Project-native centerline explorer. This is a geometric view, not MuJoCo. */
'use strict';
const canvas=document.querySelector('#airway'),ctx=canvas.getContext('2d');
const select=document.querySelector('#branch'),progress=document.querySelector('#progress');
let branches=[],angle=.25,tilt=-.3,zoom=1,drag=null,center=[0,0,0],scale=1;
function project(p){const [x,y,z]=p.map((v,i)=>v-center[i]);const a=x*Math.cos(angle)+z*Math.sin(angle),b=-x*Math.sin(angle)+z*Math.cos(angle);return [canvas.clientWidth/2+a*scale*zoom,canvas.clientHeight/2+(y*Math.cos(tilt)-b*Math.sin(tilt))*scale*zoom];}
function line(points,color,width){ctx.beginPath();points.forEach((p,i)=>{const [x,y]=project(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.lineJoin='round';ctx.stroke();}
function draw(){const dpr=window.devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*dpr);canvas.height=Math.round(canvas.clientHeight*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);if(!branches.length)return;scale=Math.min(canvas.clientWidth,canvas.clientHeight)*.76/extent;branches.forEach(b=>line(b.points,'#94b4a255',1.3));const b=branches[Number(select.value)||0];const n=Math.max(1,Math.round((b.points.length-1)*Number(progress.value)/100)+1);line(b.points.slice(0,n),'#176b52',3);const [x,y]=project(b.points[n-1]);ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fillStyle='#df854e';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();document.querySelector('#length').textContent=b.length.toFixed(1)+' mm';document.querySelector('#points').textContent=b.points.length;document.querySelector('#percentage').textContent=progress.value+'%';}
let extent=1;
fetch('assets/branches.json').then(r=>{if(!r.ok)throw Error('data');return r.json()}).then(data=>{branches=data;select.replaceChildren(...branches.map((b,i)=>new Option(b.id,String(i))));const flat=branches.flatMap(b=>b.points);const ranges=[0,1,2].map(i=>{const v=flat.map(p=>p[i]);return [Math.min(...v),Math.max(...v)]});center=ranges.map(r=>(r[0]+r[1])/2);extent=Math.max(...ranges.map(r=>r[1]-r[0]));document.querySelector('#status').textContent=`已加载 ${branches.length} 条完整路径。橙色圆点表示当前选中位置。`;draw()}).catch(()=>{document.querySelector('#status').textContent='路径数据加载失败，请通过本地 HTTP 服务或 GitHub Pages 打开页面。';select.disabled=true;});
select.addEventListener('change',draw);progress.addEventListener('input',draw);document.querySelector('#reset').onclick=()=>{angle=.25;tilt=-.3;zoom=1;draw()};
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;angle+=(e.clientX-drag[0])*.008;tilt+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.min(3,Math.max(.5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
canvas.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','='].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')angle-=.1;if(e.key==='ArrowRight')angle+=.1;if(e.key==='ArrowUp')tilt-=.1;if(e.key==='ArrowDown')tilt+=.1;if(e.key==='+'||e.key==='=')zoom=Math.min(3,zoom*1.1);if(e.key==='-')zoom=Math.max(.5,zoom/1.1);draw()};
new ResizeObserver(draw).observe(canvas.parentElement);
