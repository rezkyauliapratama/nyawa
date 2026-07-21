package server

import (
	"fmt"
	"io"
)

func (s *Server) writeDashboardHTML(w io.Writer) {
	fmt.Fprint(w, `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Nyawa Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}
.container{max-width:1200px;margin:auto}
h1{font-size:1.8rem;margin-bottom:4px;color:#58a6ff}
.sub{color:#8b949e;font-size:.9rem;margin-bottom:24px}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.card h3{font-size:.85rem;text-transform:uppercase;color:#8b949e;margin-bottom:8px}
.card .val{font-size:1.6rem;font-weight:700;color:#f0f6fc}
input,select,button{font-size:.95rem;padding:10px 14px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9}
input{flex:1;min-width:0;outline:none;width:200px}
input:focus{border-color:#58a6ff}
button{cursor:pointer;background:#21262d;font-weight:600;white-space:nowrap}
button:hover{background:#30363d}
button.primary{background:#238636;color:#fff}
button.primary:hover{background:#2ea043}
.search-bar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{text-align:left;padding:10px 12px;color:#8b949e;font-weight:600;border-bottom:1px solid #30363d}
td{padding:10px 12px;border-bottom:1px solid #21262d;vertical-align:top}
tr:hover td{background:#161b22}
td.maxw{max-width:350px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;font-weight:600}
.tag-tech{background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb44}
.tag-note{background:#23863622;color:#3fb950;border:1px solid #23863644}
.ns-badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#30363d;font-size:.75rem}
.pagination{display:flex;gap:8px;justify-content:center;margin-top:16px;align-items:center}
.empty{text-align:center;padding:40px;color:#8b949e}
.del{background:transparent;border:none;color:#da3633;cursor:pointer;padding:4px 8px}
.del:hover{background:#da363322}
.toast{position:fixed;bottom:24px;right:24px;padding:12px 20px;border-radius:8px;color:#fff;font-weight:600}
.toast.ok{background:#238636}.toast.err{background:#da3633}
textarea{width:100%;min-height:80px;padding:10px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;font-size:.95rem;resize:vertical;margin-bottom:8px;outline:none}
textarea:focus{border-color:#58a6ff}
.flex{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
</style></head>
<body><div class="container">
<h1>Nyawa</h1>
<div class="sub">Offline-First AI Memory Engine</div>
<div class="grid" id="stats"></div>
<div class="card"><h3>Store Memory</h3>
<textarea id="storeText" placeholder="Enter memory content..."></textarea>
<div class="flex"><input id="storeNS" placeholder="namespace (default)"><button class="primary" onclick="doStore()">Store</button></div></div>
<div class="card" style="margin-bottom:24px"><h3>Search</h3>
<div class="search-bar">
<input id="q" placeholder="Search memories..." onkeydown="if(event.key==='Enter')doSearch()">
<select id="nsSel"><option value="">All namespaces</option></select>
<button class="primary" onclick="doSearch()">Search</button>
<button onclick="loadList(1)">Browse All</button>
</div></div>
<div class="card">
<div style="display:flex;justify-content:space-between;margin-bottom:8px"><h3 id="listTitle">Recent Memories</h3><span id="listCount" style="color:#8b949e;font-size:.9rem"></span></div>
<table><thead><tr><th>Content</th><th>Type</th><th>NS</th><th>Score</th><th>Date</th><th></th></tr></thead><tbody id="listBody"></tbody></table>
<div id="listEmpty" class="empty" style="display:none">No memories yet.</div>
<div id="listPages" class="pagination"></div>
</div></div>
<script>
const A='';let CP=1;

async function loadStats(){try{
  var r=await fetch(A+'/v1/stats'),d=await r.json(),s=d.store||d;
  var items=[['Memories',s.total_memories||0],['Vector Indexed',s.vector_indexed||0],['Entities',s.entity_nodes||0],['Edges',s.entity_edges||0],['Namespaces',Object.keys(s.namespaces||{}).length],['Superseded',s.superseded||0]];
  var html='';for(var i=0;i<items.length;i++)html+='<div class="card"><h3>'+items[i][0]+'</h3><div class="val">'+items[i][1]+'</div></div>';
  document.getElementById('stats').innerHTML=html;
}catch(e){console.error(e)}}

async function loadNS(){try{
  var r=await fetch(A+'/v1/namespaces'),d=await r.json(),sel=document.getElementById('nsSel');
  var html='<option value="">All namespaces</option>';for(var k in d)html+='<option>'+k+'</option>';
  sel.innerHTML=html;
}catch(e){}}

async function loadList(page,ns){CP=page||1;ns=ns||document.getElementById('nsSel').value;
  try{
    var r=await fetch(A+'/v1/memories?page='+CP+'&per_page=15&ns='+encodeURIComponent(ns)),d=await r.json();
    renderMem(d.memories||[],'Recent Memories');renderPages(d.total,d.page,d.per_page);
  }catch(e){toast(e.message,'err')}}

async function doSearch(){
  var q=document.getElementById('q').value;if(!q)return loadList(1);
  var ns=document.getElementById('nsSel').value;
  try{
    var r=await fetch(A+'/v1/recall',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,namespace:ns,limit:20})}),d=await r.json();
    document.getElementById('listPages').innerHTML='';renderMem(d.results||[],'Results');
  }catch(e){toast(e.message,'err')}}

async function doStore(){
  var c=document.getElementById('storeText').value;if(!c)return;
  var ns=document.getElementById('storeNS').value||'default';
  try{
    var r=await fetch(A+'/v1/memories',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:c,namespace:ns})});
    if(!r.ok){var e=await r.json();toast(e.error||'Error','err');return}
    document.getElementById('storeText').value='';toast('Stored!','ok');loadStats();loadList(1);
  }catch(e){toast(e.message,'err')}}

async function delMem(id){
  if(!confirm('Delete?'))return;
  try{var r=await fetch(A+'/v1/forget/'+id,{method:'DELETE'});
    if(!r.ok){toast('Delete failed','err');return}
    toast('Forgotten','ok');loadStats();loadList();
  }catch(e){toast(e.message,'err')}}

function renderMem(items,title){
  document.getElementById('listTitle').textContent=title||'Memories';
  document.getElementById('listCount').textContent=items.length+' results';
  var tb=document.getElementById('listBody'),em=document.getElementById('listEmpty');
  if(!items.length){tb.innerHTML='';em.style.display='block';return}
  em.style.display='none';var h='';
  for(var i=0;i<items.length;i++){
    var m=items[i],tc=(m.type||'note').toLowerCase(),cr=(m.created_at||'').slice(0,10);
    var sc=typeof m.score==='number'?m.score.toFixed(4):'';
    h+='<tr><td class="maxw" title="'+esc(m.content)+'">'+esc(trunc(m.content,70))+'</td>'
      +'<td><span class="tag tag-'+tc+'">'+esc(m.type||'note')+'</span></td>'
      +'<td><span class="ns-badge">'+esc(m.namespace||'default')+'</span></td>'
      +'<td style="color:#8b949e">'+sc+'</td>'
      +'<td style="color:#8b949e;font-size:.85rem">'+cr+'</td>'
      +'<td><button class="del" onclick="delMem(\''+m.id+'\')">x</button></td></tr>';
  }
  tb.innerHTML=h;
}

function renderPages(total,page,perPage){
  var el=document.getElementById('listPages'),pages=Math.ceil(total/(perPage||15));
  if(pages<=1){el.innerHTML='';return}
  var h='';if(page>1)h+='<button onclick="loadList('+(page-1)+')">Prev</button>';
  h+='<span>Page '+page+' of '+pages+'</span>';
  if(page<pages)h+='<button onclick="loadList('+(page+1)+')">Next</button>';
  el.innerHTML=h;
}

function toast(msg,tp){var d=document.createElement('div');d.className='toast '+tp;d.textContent=msg;document.body.appendChild(d);setTimeout(function(){d.remove()},2500)}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function trunc(s,n){return s.length>n?s.slice(0,n-1)+'...':s}

loadStats();loadNS();loadList(1);
</script></body></html>`)
}
