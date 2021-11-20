System.register(["./chunk-vendor.js","./chunk-frameworks.js"],(function(){"use strict";var e,t,n,r,o;return{setters:[function(o){e=o._,t=o.t,n=o.b,r=o.c},function(e){o=e.a3}],execute:function(){let i=class ReadmeTocElement extends HTMLElement{connectedCallback(){var e;null===(e=this.trigger)||void 0===e||e.addEventListener("menu:activate",this.onMenuOpened.bind(this));const t=this.getHeadings();this.observer=new IntersectionObserver((()=>this.observerCallback()),{root:null,rootMargin:"0px",threshold:1});for(const n of t||[])this.observer.observe(n)}disconnectedCallback(){var e,t;null===(e=this.trigger)||void 0===e||e.removeEventListener("menu:activate",this.onMenuOpened),null===(t=this.observer)||void 0===t||t.disconnect()}blur(){window.setTimeout((()=>{document.activeElement&&document.activeElement.blur()}),0)}onMenuOpened(e){const t=e.currentTarget,n=t.getAttribute("data-menu-hydro-click")||"",r=t.getAttribute("data-menu-hydro-click-hmac")||"",i=t.getAttribute("data-hydro-client-context")||"";o(n,r,i),this.observerCallback()}getHeadings(){return this.content?this.content.querySelectorAll("h1,h2,h3,h4,h5,h6"):null}observerCallback(){const e=Array.prototype.slice.call(this.getHeadings()).filter((e=>this.isElementInViewPort(e)))[0];for(const t of this.entries||[])t.removeAttribute("aria-current"),t.style.backgroundColor="";if(e){const t=this.mapHeadingToListItemUsingAnchor(e);if(t){t.setAttribute("aria-current","page"),t.style.backgroundColor="var(--color-accent-emphasis)";const e=t.closest("div");e&&t.offsetTop&&(e.scrollTop=t.offsetTop-parseInt(getComputedStyle(e).paddingTop))}}}isElementInViewPort(e){return e.getBoundingClientRect().y>=0}mapHeadingToListItemUsingAnchor(e){const t=e.getElementsByTagName("a")[0];if(t&&this.entries)return this.entries.find((e=>e.href.replace("user-content-","")===t.href))}};e([t],i.prototype,"trigger",void 0),e([t],i.prototype,"content",void 0),e([n],i.prototype,"entries",void 0),i=e([r],i)}}}));
//# sourceMappingURL=chunk-readme-toc-element-e086b57d.js.map
