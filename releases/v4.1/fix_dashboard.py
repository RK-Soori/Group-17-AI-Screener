with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('Version 3.0', 'Version 5.0')
content = content.replace('45%', '65%')
content = content.replace('width: 45%', 'width: 65%')
content = content.replace('35%', '20%')
content = content.replace('width: 35%', 'width: 20%')
content = content.replace('5%', '0%')
content = content.replace('width: 5%', 'width: 0%')

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(content)
