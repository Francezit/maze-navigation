from .agent import StaticAgent


class InteractionStrategy:
   
   def __call__(self, agent: StaticAgent, visible_agents: list[StaticAgent], agent_costs: dict)->float:
      raise NotImplementedError("InteractionStrategy is an abstract class. Please implement the __call__ method.")
   
class StaticInteractionStrategy(InteractionStrategy):
    
    def __call__(self, agent: StaticAgent, visible_agents: list[StaticAgent], agent_costs: dict) -> float:
        """
        Versione che esegue tutti i controlli di visibilità e log, 
        ma NON modifica il beta.
        Utile per equalizzare l'overhead tra interaction() e interactionCheck().
        """
        if not visible_agents:
            return agent.get_beta  # Nessun cambiamento
        
        # Stampa i log degli agenti visibili 
        visible_agents_with_cost = [
            f"{a.name} cost:{agent_costs[a]}, beta:{a.get_beta}" 
            for a in visible_agents
        ]
        print(f"Agent {agent.name} Cost:{agent_costs[agent]} Beta:{agent.get_beta} (NO UPDATE). Visible agents: {visible_agents_with_cost}")
        
        return agent.get_beta  # Restituisce il beta originale senza modifiche

class DynamicInteractionStrategy(InteractionStrategy):

    def __call__(self, agent: StaticAgent, visible_agents: list[StaticAgent], agent_costs: dict) -> float:
            """
            Aggiorna il valore di beta dell'agente in base agli agenti visibili e ai loro costi.
            """
            if not visible_agents:
                return agent.get_beta  # Nessun agente visibile, mantieni il beta corrente

            # Calcola la somma dei prodotti tra i beta e i costi degli agenti visibili
            sum_beta_c = sum(a.get_beta * agent_costs[a] for a in visible_agents)
            
            # Aggiungi il contributo del beta e del costo dell'agente stesso
            sum_beta_c += agent.get_beta * agent_costs[agent]
            
            # Calcola la somma totale dei costi
            sum_c = sum(agent_costs[a] for a in visible_agents) + agent_costs[agent]
            
            # Calcola il nuovo beta come media ponderata
            new_beta = sum_beta_c / sum_c
            
            return new_beta 

class CompetitionInteractionStrategy(InteractionStrategy):

    def __call__(self, agent: StaticAgent, visible_agents: list[StaticAgent], agent_costs: dict, alpha: float = 1) -> float:

            if not visible_agents:
                return agent.get_beta  # Nessun agente visibile, mantieni il beta corrente

            C_self = agent_costs[agent]
            sum_positive_diff = 0.0
            sum_C = 0.0

            for a in visible_agents:
                C_i = agent_costs[a]
                sum_positive_diff += max(C_self - C_i, 0)
                sum_C += C_i

            if sum_C == 0:  # Evita divisione per zero
                return agent.get_beta

            # Calcola il nuovo beta 
            new_beta = agent.get_beta * (1 - alpha * (sum_positive_diff / sum_C))
            
            # Assicurati che beta non diventi negativo
            return max(new_beta, 0)

class ImitationInteractionStrategy(InteractionStrategy):
    
    def __call__(self, agent: StaticAgent, visible_agents: list[StaticAgent], agent_costs: dict) -> float:
        """
        Aggiorna il valore di beta dell'agente in base agli agenti visibili con beta inferiore.
        Se nessun agente visibile ha un beta minore, mantiene il valore attuale.
        """
        # Filtra solo gli agenti con beta < beta dell'agente stesso
        filtered_agents = [a for a in visible_agents if a.get_beta < agent.get_beta]

        if not filtered_agents:
            return agent.get_beta  # Nessun agente valido, mantieni il beta corrente

        # Calcola la somma dei prodotti tra i beta e i costi degli agenti filtrati
        sum_beta_c = sum(a.get_beta * agent_costs[a] for a in filtered_agents)
        
        # Aggiungi il contributo del beta e del costo dell'agente stesso
        sum_beta_c += agent.get_beta * agent_costs[agent]
        
        # Calcola la somma totale dei costi
        sum_c = sum(agent_costs[a] for a in filtered_agents) + agent_costs[agent]
        
        # Calcola il nuovo beta come media ponderata
        new_beta = sum_beta_c / sum_c
        
        return new_beta
