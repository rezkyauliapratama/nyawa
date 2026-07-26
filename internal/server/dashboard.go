package server

import (
	"fmt"
	"io"
)

func (s *Server) writeDashboardHTML(w io.Writer) {
	fmt.Fprint(w, headerHTML)
	fmt.Fprint(w, bodyHTML)
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
button.danger{background:#da3633;color:#fff}
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
.section-title{font-size:1.4rem;color:#58a6ff;margin:32px 0 16px 0;padding-top:24px;border-top:1px solid #30363d}
.slider-row{display:flex;gap:12px;align-items:center}
.slider-row input[type=range]{flex:1;background:transparent}
.slider-row span{color:#8b949e;font-size:.85rem;min-width:60px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="container" id="app"><h1>Nyawa</h1><div class="sub">Offline-First AI Memory Engine</div><div class="grid" id="stats"></div><div class="card"><h3>Store Memory</h3><textarea id="storeText" placeholder="Enter memory content..."></textarea><div class="flex"><input id="storeNS" placeholder="namespace (default)"><button class="primary" onclick="doStore()">Store</button></div></div><div class="card" style="margin-bottom:24px"><h3>Search</h3><div class="search-bar"><input id="q" placeholder="Search memories..." onkeydown="if(event.key==='Enter')doSearch()"><select id="nsSel"></select><button class="primary" onclick="doSearch()">Search</button><button onclick="loadList(1)">Browse All</button></div></div><div class="card"><div style="display:flex;justify-content:space-between;margin-bottom:8px"><h3 id="listTitle">Recent Memories</h3><span id="listCount" style="color:#8b949e;font-size:.9rem"></span></div><table><thead><tr><th>Content</th><th>Type</th><th>NS</th><th>Score</th><th>Date</th><th></th></tr></thead><tbody id="listBody"></tbody></table><div id="listEmpty" class="empty" style="display:none">No memories yet.</div><div id="listPages" class="pagination"></div></div><script>
const A='';let CP=1;
function qs(s){return document.querySelector(s)}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function trunc(s,n){return s.length>n?s.slice(0,n-1)+'...':s}
function toast(msg,tp){var d=document.createElement('div');d.className='toast '+tp;d.textContent=msg;document.body.appendChild(d);setTimeout(function(){d.remove()},2500)}
async function loadStats(){try{var r=await fetch(A+'/v1/stats'),d=await r.json(),s=d.store||d;var html='';var items=[['Memories',s.total_memories||0],['Vector Indexed',s.vector_indexed||0],['Entity Nodes',s.entity_nodes||0],['Entity Edges',s.entity_edges||0],['Namespaces',Object.keys(s.namespaces||{}).length],['Superseded',s.superseded||0]];for(var i=0;i<items.length;i++) html+='<div class="card"><h3>'+items[i][0]+'</h3><div class="val">'+items[i][1]+'</div></div>';qs('#stats').innerHTML=html}catch(e){console.error(e)}}
async function loadNS(){try{var r=await fetch(A+'/v1/namespaces'),d=await r.json(),sel=qs('#nsSel');var html='<option value="">All namespaces</option>';for(var k in d) html+='<option>'+k+'</option>';sel.innerHTML=html}catch(e){}}
async function loadList(page){CP=page||1
try{var r=await fetch(A+'/v1/memories?page='+CP+'&per_page=15'),d=await r.json();renderMem(d.memories||[],'Recent Memories');renderPages(d.total,d.page,d.per_page)}catch(e){toast(e.message,'err')}}
async function doSearch(){var q=qs('#q').value;if(!q)return loadList(1)
try{var r=await fetch(A+'/v1/recall',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,limit:20})}),d=await r.json();qs('#listPages').innerHTML='';renderMem(d.results||[],'Results: "'+esc(q)+'"')}catch(e){toast(e.message,'err')}}
async function doStore(){var c=qs('#storeText').value;if(!c)return
var ns=qs('#storeNS').value||'default'
try{var r=await fetch(A+'/v1/memories',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:c,namespace:ns})});if(!r.ok){var e=await r.json();toast(e.error||'Error','err');return}qs('#storeText').value='';toast('Stored!','ok');loadStats();loadList(1)}catch(e){toast(e.message,'err')}}
function renderMem(items,title){qs('#listTitle').textContent=title||'Memories';qs('#listCount').textContent=items.length+' results';var tb=qs('#listBody'),em=qs('#listEmpty');if(!items.length){tb.innerHTML='';em.style.display='block';return}em.style.display='none';var h='';for(var i=0;i<items.length;i++){var m=items[i];h+='<tr><td class="maxw" title="'+esc(m.content)+'">'+esc(trunc(m.content,70))+'</td><td>'+esc(m.type||'')+'</td><td>'+esc(m.namespace||'')+'</td><td>'+(typeof m.score==='number'?m.score.toFixed(4):'')+'</td><td>'+(m.created_at||'').slice(0,10)+'</td></tr>'}tb.innerHTML=h}
function renderPages(total,page,perPage){var el=qs('#listPages'),pages=Math.ceil(total/(perPage||15));if(pages<=1){el.innerHTML='';return}var h='';if(page>1)h+='<button onclick="loadList('+(page-1)+')">Prev</button>';h+='<span>Page '+page+' of '+pages+'</span>';if(page<pages)h+='<button onclick="loadList('+(page+1)+')">Next</button>';el.innerHTML=h}
loadStats();loadNS();loadList(1);
</script></div>`

const bodyHTML = `
<div class="container" style="margin-top:0;padding-top:0">
  <div class="section-title">RAG — Retrieval-Augmented Generation</div>
  <div class="grid" id="ragStats"></div>
  <div class="card" style="margin-bottom:16px">
    <h3>Collections</h3>
    <div class="flex">
      <input id="ragColName" placeholder="Collection name" style="max-width:200px">
      <input id="ragColDesc" placeholder="Description (optional)" style="max-width:300px">
      <input id="ragColCS" placeholder="Chunk size (default 500)" style="max-width:120px" type="number" value="500">
      <button class="primary" onclick="ragCreateCol()">Create</button>
    </div>
    <table><thead><tr><th>Name</th><th>Description</th><th>Chunk Size</th><th>Docs</th><th></th></tr></thead>
    <tbody id="ragColBody"><tr><td colspan="5" class="empty">No collections</td></tr></tbody></table>
  </div>
  <div class="card" style="margin-bottom:16px">
    <h3>Ingest File</h3>
    <div class="flex">
      <input id="ragFilePath" placeholder="File path (e.g. /path/to/doc.txt)" style="flex:2">
      <input id="ragIngestCol" placeholder="Collection name" style="flex:1">
      <button class="primary" onclick="ragIngest()">Ingest</button>
    </div>
  </div>
  <div class="card">
    <h3>RAG Query</h3>
    <div class="search-bar">
      <input id="ragQuery" placeholder="Ask a question about your documents..." onkeydown="if(event.key==='Enter')ragSearch()">
      <select id="ragQCol"><option value="">All collections</option></select>
      <button class="primary" onclick="ragSearch()">Search</button>
    </div>
    <div class="slider-row" style="margin-bottom:12px">
      <span>Top-K:</span>
      <input type="range" id="ragTopK" min="1" max="20" value="5" oninput="qs('#ragTopKVal').textContent=this.value">
      <span id="ragTopKVal">5</span>
    </div>
    <table><thead><tr><th>#</th><th>Content</th><th>Document</th><th>Score</th></tr></thead>
    <tbody id="ragResBody"><tr><td colspan="4" class="empty">No results yet</td></tr></tbody></table>
  </div>
</div>
<script>
async function ragLoadStats(){try{var r=await fetch(A+'/v1/rag/stats'),d=await r.json();var items=[['Collections',d.collections||0],['Documents',d.documents||0],['Chunks',d.chunks||0]];var h='';for(var i=0;i<items.length;i++) h+='<div class="card"><h3>'+items[i][0]+'</h3><div class="val">'+items[i][1]+'</div></div>';qs('#ragStats').innerHTML=h}catch(e){console.error('rag stats:',e)}}
async function ragLoadCols(){try{var r=await fetch(A+'/v1/rag/collections'),d=await r.json(),cols=d.collections||[];var tb=qs('#ragColBody'),sel=qs('#ragQCol');if(!cols.length){tb.innerHTML='<tr><td colspan="5" class="empty">No collections</td></tr>';return}var h='';for(var i=0;i<cols.length;i++){var c=cols[i];h+='<tr><td><strong>'+esc(c.name)+'</strong></td><td style="color:#8b949e">'+esc(c.description||'')+'</td><td>'+c.chunk_size+'</td><td>'+c.doc_count+'</td><td><button class="del" onclick="ragDelCol(\''+esc(c.name)+'\')">x</button></td></tr>'}tb.innerHTML=h;var oh='<option value="">All collections</option>';for(var i=0;i<cols.length;i++) oh+='<option>'+esc(cols[i].name)+'</option>';sel.innerHTML=oh}catch(e){console.error('rag cols:',e)}}
async function ragCreateCol(){try{var n=qs('#ragColName').value;if(!n){toast('Name required','err');return}var d=qs('#ragColDesc').value,cs=parseInt(qs('#ragColCS').value)||500;var r=await fetch(A+'/v1/rag/collections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n,description:d,chunk_size:cs})});if(!r.ok){var e=await r.json();toast(e.error||'Error','err');return}qs('#ragColName').value='';qs('#ragColDesc').value='';toast('Collection created','ok');ragLoadCols();ragLoadStats()}catch(e){toast(e.message,'err')}}
async function ragDelCol(name){try{if(!confirm('Delete collection "'+name+'"?'))return;var r=await fetch(A+'/v1/rag/collections/'+encodeURIComponent(name),{method:'DELETE'});if(!r.ok){toast('Delete failed','err');return}toast('Deleted','ok');ragLoadCols();ragLoadStats()}catch(e){toast(e.message,'err')}}
async function ragIngest(){try{var fp=qs('#ragFilePath').value,cn=qs('#ragIngestCol').value;if(!fp||!cn){toast('File path and collection required','err');return}var r=await fetch(A+'/v1/rag/ingest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file_path:fp,collection:cn})});if(!r.ok){var e=await r.json();toast(e.error||'Error','err');return}var d=await r.json();toast('Ingested: '+d.filename+' ('+d.chunk_count+' chunks)','ok');qs('#ragFilePath').value='';ragLoadCols();ragLoadStats()}catch(e){toast(e.message,'err')}}
async function ragSearch(){try{var q=qs('#ragQuery').value;if(!q){toast('Enter a query','err');return}var tk=parseInt(qs('#ragTopK').value)||5,c=qs('#ragQCol').value;var r=await fetch(A+'/v1/rag/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,top_k:tk,collection:c})}),d=await r.json();var res=d.results||[],tb=qs('#ragResBody');if(!res.length){tb.innerHTML='<tr><td colspan="4" class="empty">No results</td></tr>';return}var h='';for(var i=0;i<res.length;i++){var x=res[i];h+='<tr><td style="color:#8b949e;width:30px">'+(i+1)+'</td><td class="maxw" title="'+esc(x.content)+'" style="max-width:500px">'+esc(trunc(x.content,120))+'</td><td style="color:#58a6ff">'+esc(x.document||'')+'</td><td style="color:#8b949e">'+x.score.toFixed(4)+'</td></tr>'}tb.innerHTML=h}catch(e){toast(e.message,'err')}}
ragLoadStats();ragLoadCols();
</script></body></html>`
