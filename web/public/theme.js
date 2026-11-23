// Theme: runs in <head> before first paint, so the page never flashes the other theme.
// A saved choice wins; otherwise the reader's system setting, followed live.
// A plain script from this origin, because the pages' CSP forbids inline scripts.
(function () {
  var KEY = "stowaway-theme", root = document.documentElement;
  function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  var mq = window.matchMedia ? matchMedia("(prefers-color-scheme: dark)") : null;
  function apply(dark) { root.classList.toggle("dark", dark); }
  var s = saved();
  apply(s ? s === "dark" : !!(mq && mq.matches));
  if (mq && mq.addEventListener) mq.addEventListener("change", function (e) { if (!saved()) apply(e.matches); });
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.addEventListener("click", function () {
        var dark = !root.classList.contains("dark");
        apply(dark);
        try { localStorage.setItem(KEY, dark ? "dark" : "light"); } catch (e) {}
      });
    });
  });
})();
