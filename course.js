document.getElementById('skillForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var skills = document.getElementById('skills').value;
    // You can replace this with your Python code to fetch recommendations
    // For now, let's just display a message with entered skills
    document.getElementById('recommendationResult').innerHTML = '<p>Recommended courses for skills: ' + skills + '</p>';
});
