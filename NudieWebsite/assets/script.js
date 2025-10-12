document.addEventListener('DOMContentLoaded',()=>{
  const y=document.getElementById('year');
  if(y) y.textContent=new Date().getFullYear().toString();
  // Simple 18+ gate
  const GATE_KEY='nuditude_age_ok';
  if(!localStorage.getItem(GATE_KEY)){
    const modal=document.createElement('div');
    modal.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.7);display:flex;align-items:center;justify-content:center;z-index:9999;';
    modal.innerHTML=`<div style="background:#14141a;border:1px solid #2a2a35;border-radius:12px;max-width:520px;padding:18px;text-align:center;color:#e9e9ee">
      <h2 style="margin:0 0 8px">Adults Only (18+)</h2>
      <p style="color:#b6b7c2;margin:0 0 12px">By entering, you confirm you are 18+ and agree to our Terms.</p>
      <div style="display:flex;gap:8px;justify-content:center">
        <button id="age-yes" class="btn btn-primary" style="cursor:pointer">I am 18+</button>
        <a class="btn" href="https://www.google.com" style="cursor:pointer">Leave</a>
      </div>
    </div>`;
    document.body.appendChild(modal);
    const ok=modal.querySelector('#age-yes');
    ok?.addEventListener('click',()=>{localStorage.setItem(GATE_KEY,'1');modal.remove();});
  }
});
