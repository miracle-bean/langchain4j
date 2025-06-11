package dev.langchain4j.agentic;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.service.V;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static dev.langchain4j.internal.Utils.isNullOrBlank;

public record AgentSpecification(Method method, String name, String description, List<String> arguments) {

    public static AgentSpecification fromMethod(Method method) {
        Agent annotation = method.getAnnotation(Agent.class);
        String name = isNullOrBlank(annotation.name()) ? method.getName() : annotation.name();
        String description = String.join("\n", annotation.value());
        List<String> arguments = Stream.of(method.getParameters())
                .map(AgentSpecification::parameterName)
                .toList();
        return new AgentSpecification(method, name, description, arguments);
    }

    public String toCard() {
        return "{" + name + ": " + description + ", " + arguments + "}";
    }

    public Object[] toInvocationArguments(Map<String, String> arguments) {
        Parameter[] parameters = method.getParameters();
        Object[] invocationArgs = new Object[parameters.length];
        int i = 0;
        for (Parameter parameter : parameters) {
            String argName = parameterName(parameter);
            String argValue = arguments.get(argName);
            if (argValue == null) {
                throw new IllegalArgumentException("Missing argument: " + argName);
            }
            invocationArgs[i++] = parseArgument(argValue, parameter.getType());
        }
        return invocationArgs;
    }

    private Object parseArgument(String argValue, Class<?> type) {
        return switch (type.getName()) {
            case "java.lang.String" -> argValue;
            case "int", "java.lang.Integer" -> Integer.parseInt(argValue);
            case "long", "java.lang.Long" -> Long.parseLong(argValue);
            case "double", "java.lang.Double" -> Double.parseDouble(argValue);
            case "boolean", "java.lang.Boolean" -> Boolean.parseBoolean(argValue);
            default -> throw new IllegalArgumentException("Unsupported type: " + type);
        };
    }

    private static String parameterName(Parameter parameter) {
        P p = parameter.getAnnotation(P.class);
        if (p != null) {
            return p.value();
        }
        V v = parameter.getAnnotation(V.class);
        if (v != null) {
            return v.value();
        }
        if (parameter.isNamePresent()) {
            return parameter.getName();
        } else {
            throw new IllegalArgumentException("Parameter name not specified and no @P or @V annotation present: " + parameter);
        }
    }
}
