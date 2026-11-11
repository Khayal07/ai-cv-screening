// Loaded in <head> without defer, so a saved theme applies before the first paint.
(function applySavedTheme() {
  try {
    var savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light" || savedTheme === "dark") {
      document.documentElement.setAttribute("data-theme", savedTheme);
    }
  } catch (e) {}
})();
