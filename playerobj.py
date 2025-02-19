
class Player:
    def __init__(self, slug, name, team, positions):
        self.slug = slug
        self.name = name
        self.team = team
        self.positions = positions

    def __repr__(self):
        return f'{self.name} - {self.team} - {self.positions}'
    
    def name_as_file(self):
         name_file = self.name.strip()
         name_file = name_file.replace(" ", "_")
         name_file = name_file.replace("-", "_")
         name_file = name_file.replace("'", "")
         return name_file