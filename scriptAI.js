// JavaScript to handle dropdown visibility
document.addEventListener("DOMContentLoaded", function() {
    var featuresDropdown = document.getElementById("features-dropdown");
    var dropbtn = featuresDropdown.querySelector(".dropbtn");
    var dropdownContent = featuresDropdown.querySelector(".dropdown-content");

    featuresDropdown.addEventListener("mouseenter", function() {
        dropdownContent.style.display = "block";
    });

    featuresDropdown.addEventListener("mouseleave", function() {
        dropdownContent.style.display = "none";
    });

    dropbtn.addEventListener("mouseleave", function() {
        dropdownContent.style.display = "none";
    });
});
