package server

import (
	"fmt"
	"io"
)

func (s *Server) writeDashboardHTML(w io.Writer) {
	fmt.Fprint(w, headerHTML)
}

const headerHTML = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Nyawa Dashboard</title><style>
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
input{flex:1;min-width:0;outline:none}
input:focus{border-color:#58a6ff}
button{cursor:pointer;background:#21262d;font-weight:600;white-space:nowrap}
button:hover{background:#30363d}
button.primary{background:#238636;border-color:#238636;color:#fff}
button.primary:hover{background:#2ea043}
.search-bar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{text-align:left;padding:10px 12px;color:#8b949e;font-weight:600;border-bottom:1px solid #30363d}
td{padding:10px 12px;border-bottom:1px solid #21262d;vertical-align:top}
tr:hover td{background:#161b22}
td.maxw{max-width:350px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;font-weight:600}
.ns-badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#30363d;font-size:.75rem}
.pagination{display:flex;gap:8px;justify-content:center;margin-top:16px;align-items:center}
.pagination span{padding:6px 12px;color:#8b949e}
.empty{text-align:center;padding:40px;color:#8b949e}
textarea{width:100%;min-height:80px;padding:10px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;font-size:.95rem;resize:vertical;margin-bottom:8px;outline:none}
textarea:focus{border-color:#58a6ff}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="container" id="app"><h1>Nyawa</h1><div class="sub">Offline-First AI Memory Engine</div><div class="grid" id="stats"></div><div class="card"><h3>Store Memory</h3><textarea id="storeText" placeholder="Enter memory content..."></textarea><div class="flex"><input id="storeNS" placeholder="namespace (default)"><button class="primary" onclick="doStore()">Store</button></div></div><div class="card" style="margin-bottom:24px"><h3>Search</h3><div class="search-bar"><input id="q" placeholder="Search memories..." onkeydown="if(event.key==='Enter')doSearch()"><select id="nsSel"></select><button class="primary" onclick="doSearch()">Search</button><button onclick="loadList(1)">Browse All</button></div></div><div class="card"><div style="display:flex;justify-content:space-between;margin-bottom:8px"><h3 id="listTitle">Recent Memories</h3><span id="listCount" style="color:#8b949e;font-size:.9rem"></span></div><table><thead><tr><th>Content</th><th>Type</th><th>NS</th><th>Score</th><th>Date</th><th></th></tr></thead><tbody id="listBody"></tbody></table><div id="listEmpty" class="empty" style="display:none">No memories yet.</div><div id="listPages" class="pagination"></div></div></div><script>
const A='';let CP=1;
function qs(s){return document.querySelector(s)}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function trunc(s,n){return s.length>n?s.slice(0,n-1)+'...':s}
function toast(msg,tp){var d=document.createElement('div');d.className='toast '+tp;d.textContent=msg;document.body.appendChild(d);setTimeout(function(){d.remove()},2500)}
async function loadStats(){try{var r=await fetch(A+'/v1/stats'),d=await r.json(),s=d.store||d;qs('#stats').innerHTML='<div class="card"><h3>Memories</h3><div class="val">'+(s.total_memories||0)+'</div></div><div class="card"><h3>Vector Indexed</h3><div class="val">'+(s.vector_indexed||0)+'</div></div><div class="card"><h3>Entity Nodes</h3><div class="val">'+(s.entity_nodes||0)+'</div></div>'}catch(e){console.error(e)}}
async function loadList(page){CP=page||1
try{var r=await fetch(A+'/v1/memories?page='+CP+'&per_page=15'),d=await r.json();renderMem(d.memories||[],'Recent Memories');renderPages(d.total,d.page,d.per_page)}catch(e){toast(e.message,'err')}}
async function doSearch(){var q=qs('#q').value;if(!q)return loadList(1)
try{var r=await fetch(A+'/v1/recall',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,limit:20})}),d=await r.json();qs('#listPages').innerHTML='';renderMem(d.results||[],'Results: "'+esc(q)+'"')}catch(e){toast(e.message,'err')}}
async function doStore(){var c=qs('#storeText').value;if(!c)return
var ns=qs('#storeNS').value||'default'
try{var r=await fetch(A+'/v1/memories',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:c,namespace:ns})});if(!r.ok){var e=await r.json();toast(e.error||'Error','err');return}qs('#storeText').value='';toast('Stored!','ok');loadStats();loadList(1)}catch(e){toast(e.message,'err')}}
function renderMem(items,title){qs('#listTitle').textContent=title||'Memories';qs('#listCount').textContent=items.length+' results';var tb=qs('#listBody'),em=qs('#listEmpty');if(!items.length){tb.innerHTML='';em.style.display='block';return}em.style.display='none';var h='';for(var i=0;i<items.length;i++){var m=items[i];h+='<tr><td class="maxw" title="'+esc(m.content)+'">'+esc(trunc(m.content,70))+'</td><td>'+esc(m.type||'')+'</td><td>'+esc(m.namespace||'')+'</td><td>'+(typeof m.score==='number'?m.score.toFixed(4):'')+'</td><td>'+(m.created_at||'').slice(0,10)+'</td></tr>'}tb.innerHTML=h}
function renderPages(total,page,perPage){var el=qs('#listPages'),pages=Math.ceil(total/(perPage||15));if(pages<=1){el.innerHTML='';return}var h='';if(page>1)h+='<button onclick="loadList('+(page-1)+')">Prev</button>';h+='<span>Page '+page+' of '+pages+'</span>';if(page<pages)h+='<button onclick="loadList('+(page+1)+')">Next</button>';el.innerHTML=h}
loadStats();loadList(1);
</script></body></html>`
