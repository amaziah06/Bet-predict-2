const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>(x*100).toFixed(1)+'%';
function localDate(){const d=new Date();return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');}
function addDays(iso,n){const d=new Date(iso+'T12:00:00');d.setDate(d.getDate()+n);return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');}
function prettyDay(iso){try{return new Date(iso+'T12:00:00').toLocaleDateString([], {weekday:'long',day:'numeric',month:'short'})}catch(e){return iso}}
async function api(path,opts){
  const r=await fetch(path,Object.assign({credentials:'same-origin',headers:{'Content-Type':'application/json'}},opts||{}));
  if(r.status===401){location.href='/login';throw new Error('Please log in.');}
  const t=await r.text();let j=null;try{j=JSON.parse(t)}catch(e){}
  if(!r.ok){let d=j&&j.detail;if(Array.isArray(d))d='Please check what you typed.';throw new Error(d||('Server error '+r.status));}
  return j;
}
async function logout(){try{await fetch('/api/logout',{method:'POST',credentials:'same-origin'})}catch(e){}location.href='/login';}
function clockTime(iso){try{return new Date(iso).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}catch(e){return ''}}
/* team crest with initials fallback */
function initials(n){const w=String(n||'?').replace(/[^A-Za-zÀ-ÿ ]/g,'').trim().split(/\s+/);return ((w[0]||'?')[0]+(w.length>1?w[1][0]:(w[0]||'?')[1]||'')).toUpperCase();}
function crest(url,name,size){const c='crest '+(size||'');const i='init '+(size||'');
  if(!url)return `<span class="${i}">${esc(initials(name))}</span>`;
  return `<img class="${c}" src="${esc(url)}" alt="" loading="lazy" onerror="this.outerHTML='<span class=&quot;${i}&quot;>${esc(initials(name))}</span>'">`;}
let _tt;function toast(msg){let t=$('#toast');if(!t){t=document.createElement('div');t.id='toast';document.body.appendChild(t);}t.textContent=msg;t.classList.add('show');clearTimeout(_tt);_tt=setTimeout(()=>t.classList.remove('show'),2400);}
function skeleton(n){return Array.from({length:n||3},()=>'<div class="sk"></div>').join('');}
const ICONS={
 games:'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7l4 3-1.5 5h-5L8 10z"/></svg>',
 surest:'<svg viewBox="0 0 24 24"><path d="M12 3l2.6 5.6 6.1.7-4.5 4.2 1.2 6L12 16.5 6.6 19.5l1.2-6L3.3 9.3l6.1-.7z"/></svg>',
 tables:'<svg viewBox="0 0 24 24"><path d="M4 20V10M10 20V4M16 20v-8M22 20H2"/></svg>',
 slip:'<svg viewBox="0 0 24 24"><path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6"/></svg>',
 accuracy:'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 3v3M12 18v3"/></svg>'};
const TABS=[['games','/games','Games'],['surest','/surest','Surest'],['tables','/tables','Tables'],['slip','/slip','Slip'],['accuracy','/accuracy','Accuracy']];
function setSlipBadge(n){document.querySelectorAll('.slipbadge').forEach(b=>{b.textContent=n;b.style.display=n>0?'block':'none';});}
async function mountNav(active){
  const links=cls=>TABS.map(([k,u,l])=>`<a href="${u}" class="${active===k?'active':''}">${cls==='bot'?ICONS[k]:''}${l}${k==='slip'?'<span class="badge slipbadge" style="display:none"></span>':''}</a>`).join('');
  $('#top').innerHTML=`<div class="bar-in"><a class="brand" href="/games"><img src="/static/logo.svg" alt=""><b>Bet<span>Predict</span></b></a>
   <nav class="tabs-top">${links('top')}</nav><span class="sp"></span>
   <div class="menu" id="menu"><button class="avatar" id="av" aria-label="Account"></button><div class="menu-list"><div id="who"></div><button onclick="logout()">Log out</button></div></div></div>`;
  if(!$('#tabbar')){const n=document.createElement('nav');n.id='tabbar';n.className='tabbar';document.body.appendChild(n);}
  $('#tabbar').innerHTML=links('bot');
  $('#av').onclick=e=>{e.stopPropagation();$('#menu').classList.toggle('open');};
  document.addEventListener('click',()=>$('#menu')&&$('#menu').classList.remove('open'));
  const me=await api('/api/me');$('#who').textContent=me.email;$('#av').textContent=(me.email||'?')[0].toUpperCase();setSlipBadge(me.slip_open||0);return me;
}
async function addToSlip(fixtureId,date,market,key){
  try{const r=await api('/api/slip',{method:'POST',body:JSON.stringify({fixture_id:fixtureId,date,market,key})});setSlipBadge(r.open);toast('Added: '+r.label);}
  catch(e){toast(e.message);}
}
